import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "acac"
sys.path.insert(0, str(SOURCE_ROOT))
from scripts import acac_main  # noqa: E402


class StopAfterSeeding(Exception):
    pass


class SeedZeroTests(unittest.TestCase):
    def test_main_initializes_python_numpy_and_torch_with_zero(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            expected = (random.random(), np.random.rand(), torch.rand(1).item())

            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            with patch.object(acac_main, "get_env_params", side_effect=StopAfterSeeding):
                with self.assertRaises(StopAfterSeeding):
                    acac_main.main(SimpleNamespace(seed=0))

            actual = (random.random(), np.random.rand(), torch.rand(1).item())
            self.assertEqual(actual, expected)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)


if __name__ == "__main__":
    unittest.main()
