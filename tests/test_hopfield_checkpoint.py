"""Disk checkpoint coverage for the fixed-bank training example."""

import math
from pathlib import Path
import runpy

import numpy as np


def test_restored_fixed_bank_matches_uninterrupted_training(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "hopfield_fixed_bank.py"
    example = runpy.run_path(str(script))
    checkpoint = tmp_path / "fixed-bank"

    # The example asserts equality of the next inference state, parameter
    # update, optimizer state, and batch metrics against continuous training.
    metrics = example["run_example"](checkpoint)
    assert math.isfinite(metrics["energy"])

    restored = example["restore_checkpoint"](checkpoint)
    assert restored["next_epoch"] == 1
    bank = restored["node_buffers"]["hidden"]
    np.testing.assert_array_equal(bank["hopfield_active_mask"], [True, False, True])
    assert bank["hopfield_active_mask"].dtype == np.bool_
    np.testing.assert_array_equal(
        bank["hopfield_memory"], [[1.0, 0.25], [4.0, -2.0], [-0.5, 0.75]]
    )
