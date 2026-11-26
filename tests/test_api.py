from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import server
from core.job_store import JobStore
from core.models import (
    ClipCandidate,
    ClipOutput,
    JobRequest,
    PipelineResult,
    ReframeMode,
    ReframePlan,
    RenderResult,
)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_store = server.store
        server.store = JobStore(self.temporary.name)
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        server.store = self.previous_store
        self.temporary.cleanup()

    def _successful_pipeline(self, request: JobRequest, **_kwargs: object) -> PipelineResult:
        job_id = request.job_id
        output_dir = request.output_dir
        output_path = Path(output_dir) / "clip-01-010.00-Test-hook.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mock mp4")
        candidate = ClipCandidate(
            start=10,
            end=40,
            virality_score=91,
            hook_title="Test hook",
            reason="A strong moment.",
            suggested_caption="Watch this.",
        )
        plan = ReframePlan(
            requested_mode=ReframeMode.FIT_BLUR,
            mode=ReframeMode.FIT_BLUR,
            source_width=1920,
            source_height=1080,
            output_width=1080,
            output_height=1920,
            filter_complex="[0:v]scale=1080:1920[vout]",
        )
        return PipelineResult(
            job_id=job_id,
            status="completed",
            clips=[
                ClipOutput(
                    candidate=candidate,
                    render=RenderResult(path=output_path, encoder="libx264", duration=30),
                    reframe=plan,
                )
            ],
        )

    def test_production_ui_and_spa_routes_do_not_shadow_api_or_docs(self) -> None:
        if not (server.WEB_DIST / "index.html").is_file():
            self.skipTest("web/dist has not been built")
        homepage = self.client.get("/")
        self.assertEqual(homepage.status_code, 200)
        self.assertIn("FrameForge", homepage.text)
        deep_link = self.client.get("/studio/settings/providers")
        self.assertEqual(deep_link.status_code, 200)
        self.assertIn("<div id=\"root\"></div>", deep_link.text)
        self.assertEqual(self.client.get("/api/health").json()["status"], "ok")
        api_miss = self.client.get("/api/not-a-route")
        self.assertEqual(api_miss.status_code, 404)
        self.assertNotIn('<div id="root"></div>', api_miss.text)
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_health_runtime_and_job_flow(self) -> None:
        self.assertEqual(self.client.get("/api/health").json()["pipeline"], "ready")
        runtime = self.client.get("/api/runtime")
        self.assertEqual(runtime.status_code, 200)
        self.assertIn("encoder", runtime.json())

        with patch("api.server.run_pipeline", side_effect=self._successful_pipeline):
            submitted = self.client.post(
                "/api/jobs",
                json={"source": "https://youtu.be/abcdefghijk", "clip_count": 1, "api_key": "never-persist-this"},
            )
        self.assertEqual(submitted.status_code, 202)
        job_id = submitted.json()["job_id"]
        self.assertEqual(submitted.json()["status"], "queued")
        current = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["status"], "completed")
        self.assertEqual(current.json()["clips"][0]["encoder"], "libx264")
        downloaded = self.client.get(current.json()["clips"][0]["download_url"])
        self.assertEqual(downloaded.content, b"mock mp4")
        persisted = (Path(self.temporary.name) / job_id / "job.json").read_text(encoding="utf-8")
        self.assertNotIn("never-persist-this", persisted)

    def test_provider_settings_save_and_test_redacts_keys_and_preserves_them_on_test_failure(self) -> None:
        settings_root = Path(self.temporary.name) / "config"
        api_key = "sk-local-provider-test-secret"
        with patch.dict(os.environ, {"FRAMEFORGE_DATA_DIR": str(settings_root), "OPENAI_API_KEY": ""}):
            with patch("api.server.test_provider_connection", return_value="Connected to Openai (gpt-test).") as probe:
                response = self.client.post(
                    "/api/settings/test",
                    json={
                        "provider": "openai",
                        "model": "gpt-test",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": api_key,
                    },
                )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["ok"])
            self.assertNotIn(api_key, response.text)
            self.assertEqual(probe.call_args.kwargs["api_key"], api_key)
            public = self.client.get("/api/settings")
            self.assertTrue(public.json()["providers"]["openai"]["configured"])
            self.assertNotIn(api_key, public.text)

            with patch(
                "api.server.test_provider_connection",
                side_effect=RuntimeError(f"provider echoed {api_key}, but is offline"),
            ):
                failed = self.client.post(
                    "/api/settings/test",
                    json={
                        "provider": "openai",
                        "model": "gpt-test",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "",
                    },
                )
            self.assertEqual(failed.status_code, 200)
            self.assertFalse(failed.json()["ok"])
            self.assertIn("[redacted]", failed.json()["message"])
            self.assertNotIn(api_key, failed.text)
            stored = (settings_root / "settings.json").read_text(encoding="utf-8")
            self.assertIn(api_key, stored)

    def test_provider_settings_reject_credential_bearing_base_urls(self) -> None:
        settings_root = Path(self.temporary.name) / "config"
        with patch.dict(os.environ, {"FRAMEFORGE_DATA_DIR": str(settings_root)}):
            response = self.client.post(
                "/api/settings/test",
                json={
                    "provider": "openai",
                    "model": "gpt-test",
                    "base_url": "https://user:secret@provider.example/v1",
                    "api_key": "",
                },
            )
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("secret", response.text)
        self.assertFalse((settings_root / "settings.json").exists())

    def test_delete_clip_removes_video_caption_and_job_entry(self) -> None:
        with patch("api.server.run_pipeline", side_effect=self._successful_pipeline):
            submitted = self.client.post(
                "/api/jobs",
                json={"source": "https://youtu.be/abcdefghijk", "clip_count": 1},
            )
        job_id = submitted.json()["job_id"]
        clip = self.client.get(f"/api/jobs/{job_id}").json()["clips"][0]
        video_path = Path(self.temporary.name) / job_id / "exports" / clip["file_name"]
        caption_path = Path(self.temporary.name) / job_id / "captions" / f"{video_path.stem}.ass"
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text("ASS caption sidecar", encoding="utf-8")

        deleted = self.client.delete(f"/api/clips/{clip['clip_id']}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertTrue(deleted.json()["deleted"])
        self.assertFalse(video_path.exists())
        self.assertFalse(caption_path.exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").json()["clips"], [])
        persisted = (Path(self.temporary.name) / job_id / "job.json").read_text(encoding="utf-8")
        self.assertIn('"clips": []', persisted)
        self.assertNotIn(clip["file_name"], persisted)
        self.assertEqual(self.client.delete("/api/clips/not-a-valid-token").status_code, 404)

    def test_multipart_upload_is_saved_in_job_local_storage(self) -> None:
        with patch("api.server.run_pipeline", side_effect=self._successful_pipeline):
            response = self.client.post(
                "/api/jobs/upload",
                data={"provider": "ollama", "layout": "fit_blur", "clip_count": "1"},
                files={"file": ("sample.mp4", b"uploaded local video", "video/mp4")},
            )
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        source_file = Path(self.temporary.name) / job_id / "uploads" / "source.mp4"
        self.assertEqual(source_file.read_bytes(), b"uploaded local video")
        completed = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["source_label"], "sample.mp4")

    def test_rejects_unsafe_job_id_and_unsupported_upload_extension(self) -> None:
        invalid = self.client.post("/api/jobs", json={"source": "x.mp4", "job_id": "../bad"})
        self.assertEqual(invalid.status_code, 422)
        upload = self.client.post(
            "/api/jobs/upload",
            files={"file": ("readme.txt", b"not video", "text/plain")},
        )
        self.assertEqual(upload.status_code, 415)


if __name__ == "__main__":
    unittest.main()
