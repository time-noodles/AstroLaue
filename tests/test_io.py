"""Rigaku sLaue CSV 入出力モジュール (io.py) の単体テスト."""

import glob
from pathlib import Path
import numpy as np
import pytest

from astrolaue.io import load_slaue_csv, SlaueRawData


def test_load_slaue_csv():
    csv_files = sorted(glob.glob("share/slaue-data/**/*.CSV", recursive=True))
    if not csv_files:
        pytest.skip("sLaue CSV データが存在しないためスキップ")

    sample_csv = csv_files[0]
    raw = load_slaue_csv(sample_csv)

    assert isinstance(raw, SlaueRawData)
    assert raw.polar_data.ndim == 2
    assert raw.polar_data.shape[0] == len(raw.radii_mm)
    assert raw.polar_data.shape[1] == len(raw.thetas_deg)
    assert raw.camera_length_mm > 20.0  # 通常 27.0 mm 付近

    norm = raw.to_normalized_polar()
    assert norm.shape == raw.polar_data.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
