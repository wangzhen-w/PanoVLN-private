"""Scene discovery and preservation when joining existing collections."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dataset_create.trajectory.hm3d import discover_scenes, resolve_scene_path
from dataset_create.trajectory.merge import merge_collections
from dataset_create.trajectory.pipeline import build_parser


class CollectionTests(unittest.TestCase):
    def test_all_source_folders_feed_one_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group, key in (("train", "00001-alpha"), ("val", "00800-beta")):
                folder = root / group / key
                folder.mkdir(parents=True)
                short = key.split("-", 1)[1]
                (folder / f"{short}.basis.glb").touch()
                (folder / f"{short}.basis.navmesh").touch()
            found = discover_scenes(root)
            self.assertEqual(len(found), 2)
            self.assertTrue(any("/train/" in r["scene_id"] for r in found))
            self.assertTrue(any("/val/" in r["scene_id"] for r in found))
            self.assertTrue(all("split" not in r for r in found))
            for record in found:
                self.assertEqual(resolve_scene_path(root, record["scene_id"]), Path(record["glb_path"]))
            self.assertEqual(discover_scenes(root, ["beta"])[0]["scene_key"], "00800-beta")
            with self.assertRaises(FileNotFoundError):
                discover_scenes(root, ["missing"])

    def test_collector_has_no_dataset_split_option(self):
        args = build_parser().parse_args(["collect", "--scene-root", "/scenes", "--output-root", "/data/trajectory"])
        self.assertFalse(hasattr(args, "split"))
        self.assertEqual(args.dataset_name, "trajectories")

    def collections(self):
        datasets, reports = [], []
        for group in ("train", "val"):
            scene = f"hm3d/{group}/scene-{group}/{group}.basis.glb"
            episode = {"trajectory_id": group, "scene_id": scene,
                       "metrics": {"decision_event_count": 1, "region_count": 2, "length_m": 8.},
                       "action_ids": [1, 2, 0], "final_position": [0., 0., 1.]}
            datasets.append({"schema_version": "example", "metadata": {"split": group, "dataset_name": group,
                             "scene_count": 1, "action_parameters": {"forward_step_size": .25}}, "episodes": [episode]})
            reports.append({"trajectories": 1, "scene_count": 1, "regions": 2, "connections": 1,
                            "decision_relations": 1, "route_candidates": 1, "route_families": 1,
                            "scenes": [{"scene_id": scene, "trajectories": 1, "diagnosis": "healthy"}]})
        return datasets, reports

    @patch("dataset_create.trajectory.merge.validate_dataset")
    def test_merge_preserves_episodes_and_combines_statistics(self, validate):
        datasets, reports = self.collections()
        originals = copy.deepcopy(datasets)
        result, summary = merge_collections(datasets, reports)
        self.assertEqual(result["episodes"], [d["episodes"][0] for d in originals])
        self.assertEqual(datasets, originals)
        self.assertNotIn("split", result["metadata"])
        self.assertEqual(result["metadata"]["purpose"], "training")
        self.assertEqual(summary["scene_count"], 2)
        self.assertEqual(summary["trajectories"], 2)
        self.assertEqual(summary["decision_event_histogram"], {"1": 2})
        self.assertNotIn("independent_replay", summary)

    @patch("dataset_create.trajectory.merge.validate_dataset")
    def test_merge_rejects_incompatible_parameters_or_overlapping_scenes(self, validate):
        datasets, reports = self.collections()
        datasets[1]["metadata"]["action_parameters"]["forward_step_size"] = .5
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            merge_collections(datasets, reports)
        datasets, reports = self.collections()
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge_collections([datasets[0], datasets[0]], [reports[0], reports[0]])


if __name__ == "__main__":
    unittest.main()
