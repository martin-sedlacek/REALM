import unittest
import pandas as pd
from realm_viewer.dashboard_utils import get_completed_experiments, filter_videos, filter_dataframe

class TestDashboardUtils(unittest.TestCase):
    def test_get_completed_experiments(self):
        # Create a dummy dataframe
        data = {
            'task': ['put_green_block_into_bowl', 'put_green_block_into_bowl', 'rotate_marker'],
            'perturbation': ["['Default']", "['Default']", "['V-AUG']"]
        }
        df = pd.DataFrame(data)

        # Test with repeats=2
        completed = get_completed_experiments(df, required_repeats=2)
        self.assertIn(('put_green_block_into_bowl', 'Default'), completed)
        self.assertNotIn(('rotate_marker', 'V-AUG'), completed)

        # Test with repeats=1
        completed = get_completed_experiments(df, required_repeats=1)
        self.assertIn(('put_green_block_into_bowl', 'Default'), completed)
        self.assertIn(('rotate_marker', 'V-AUG'), completed)

    def test_filter_videos(self):
        videos = [
            "/path/to/2024_rollout_put_green_block_into_bowl_Default_0.mp4",
            "/path/to/2024_rollout_put_green_block_into_bowl_V-AUG_0.mp4",
            "/path/to/2024_rollout_rotate_marker_Default_0.mp4"
        ]

        # Test empty filters (should return all)
        self.assertEqual(filter_videos(videos, [], []), videos)

        # Test filtering by task only
        filtered = filter_videos(videos, selected_tasks=['put_green_block_into_bowl'], selected_perts=[])
        self.assertEqual(len(filtered), 2)
        self.assertIn(videos[0], filtered)
        self.assertIn(videos[1], filtered)
        self.assertNotIn(videos[2], filtered)

        # Test filtering by perturbation only
        filtered = filter_videos(videos, selected_tasks=[], selected_perts=['Default'])
        self.assertEqual(len(filtered), 2)
        self.assertIn(videos[0], filtered)
        self.assertIn(videos[2], filtered)
        self.assertNotIn(videos[1], filtered)

        # Test filtering by both
        filtered = filter_videos(videos, selected_tasks=['put_green_block_into_bowl'], selected_perts=['Default'])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], videos[0])

    def test_filter_dataframe(self):
        data = {
            'task': ['put_green_block_into_bowl', 'put_green_block_into_bowl', 'rotate_marker'],
            'perturbation': ["['Default']", "['V-AUG']", "['Default']"],
            'success': [1, 0, 1]
        }
        df = pd.DataFrame(data)

        # Test empty filters (should return all)
        filtered = filter_dataframe(df, [], [])
        self.assertEqual(len(filtered), 3)

        # Test filtering by task
        filtered = filter_dataframe(df, selected_tasks=['put_green_block_into_bowl'], selected_perts=[])
        self.assertEqual(len(filtered), 2)
        self.assertTrue(all(filtered['task'] == 'put_green_block_into_bowl'))

        # Test filtering by perturbation
        filtered = filter_dataframe(df, selected_tasks=[], selected_perts=['Default'])
        self.assertEqual(len(filtered), 2)
        # Check original format is preserved in column
        self.assertTrue(any(filtered['task'] == 'put_green_block_into_bowl'))
        self.assertTrue(any(filtered['task'] == 'rotate_marker'))

        # Test filtering by both
        filtered = filter_dataframe(df, selected_tasks=['put_green_block_into_bowl'], selected_perts=['V-AUG'])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['task'], 'put_green_block_into_bowl')
        self.assertEqual(filtered.iloc[0]['perturbation'], "['V-AUG']")

if __name__ == '__main__':
    unittest.main()
