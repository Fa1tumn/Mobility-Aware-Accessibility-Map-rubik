"""Checkpoint compatibility tests using the bundled models; no hardware/network."""
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from surface_model import SurfaceModel, TorchBackend


class UnapprovedMetadata:
    pass


class CheckpointTests(unittest.TestCase):
    def test_bundled_checkpoints_load_and_run(self):
        before = torch.serialization.get_safe_globals().copy()
        model = SurfaceModel.from_checkpoints()
        try:
            result = model.analyze(np.zeros((360, 640, 3), dtype=np.uint8))
            self.assertIn(result["surface_label"],
                          model.spec["models"]["material"]["labels"] + ["unknown"])
            for key in ("area_accept_ratio", "material_accept_ratio"):
                self.assertGreaterEqual(result[key], 0)
                self.assertLessEqual(result[key], 1)
        finally:
            model.close()
        self.assertEqual(torch.serialization.get_safe_globals(), before)

    def test_unapproved_checkpoint_global_still_rejected(self):
        before = torch.serialization.get_safe_globals().copy()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "unapproved.pt"
            torch.save({"metadata": UnapprovedMetadata()}, checkpoint)
            with self.assertRaisesRegex(pickle.UnpicklingError, "Unsupported global"):
                TorchBackend(checkpoint, [], "area")
        self.assertEqual(torch.serialization.get_safe_globals(), before)


if __name__ == "__main__":
    unittest.main()
