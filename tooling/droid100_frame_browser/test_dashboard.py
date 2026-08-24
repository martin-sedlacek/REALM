import tempfile
import unittest
from pathlib import Path

from tooling.droid100_frame_browser.core import discover_frames, task_rank


class FrameBrowserTest(unittest.TestCase):
    def test_discovers_jpg_and_jpeg_in_rank_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task, panel in (("010_tenth", "panel.jpeg"), ("002_second", "panel.jpg")):
                path = root / "run_a" / "frames" / task / panel
                path.parent.mkdir(parents=True)
                path.touch()
            frames = discover_frames(root)
        self.assertEqual([frame.task for frame in frames], ["002_second", "010_tenth"])

    def test_rank_parser_tolerates_non_ranked_directories(self):
        self.assertEqual(task_rank("006_remove_lid"), 6)
        self.assertIsNone(task_rank("smoke_task"))


if __name__ == "__main__":
    unittest.main()
