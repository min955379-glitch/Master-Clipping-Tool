from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.job_store import JobStore
from core.models import PipelineResult


class JobStoreTests(unittest.TestCase):
    def test_job_snapshots_are_persisted_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JobStore(temporary)
            created = store.create("job-12345678", "sample.mp4")
            self.assertEqual(created.status, "queued")
            store.update("job-12345678", status="running", progress=0.4, stage="INGEST", message="Reading media")
            result = PipelineResult(job_id="job-12345678", status="completed")
            store.finish("job-12345678", result)

            restarted = JobStore(temporary)
            restored = restarted.get("job-12345678")
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.status, "completed")
            self.assertEqual(restored.progress, 1)
            self.assertEqual(restored.result, result)
            self.assertTrue((Path(temporary) / "job-12345678" / "job.json").is_file())

    def test_inflight_jobs_are_marked_interrupted_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JobStore(temporary)
            store.create("job-87654321", "remote video")
            store.update("job-87654321", status="running", stage="RENDER", progress=0.7)
            restored = JobStore(temporary).get("job-87654321")
            self.assertIsNotNone(restored)
            assert restored is not None
            self.assertEqual(restored.status, "failed")
            self.assertIn("interrupted", restored.message)
            self.assertIsNotNone(restored.updated_at)

    def test_job_ids_are_path_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = JobStore(temporary)
            with self.assertRaises(ValueError):
                store.create("../escape", "unsafe")
            with self.assertRaises(ValueError):
                store.work_dir("../escape")


if __name__ == "__main__":
    unittest.main()
