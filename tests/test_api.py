from __future__ import annotations

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
