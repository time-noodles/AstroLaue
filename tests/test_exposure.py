"""露光時間最適化＆ピーク飽和解析モジュール (exposure.py) の単体テスト."""

import glob
from pathlib import Path
import pytest
import numpy as np

from astrolaue.io import load_slaue_csv, get_exposure_time_from_asts, ExposureMetadata
from astrolaue.exposure import (
    analyze_exposure_and_optimize,
    SaturationStatus,
    ExposureOptimizationResult,
)
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig


def test_asts_exposure_time_extraction():
    """ASTS アーカイブから照射時間が正しく抽出できることを検証."""
    csv_files = sorted(glob.glob("share/slaue-data/**/*.CSV", recursive=True))
    if not csv_files:
        pytest.skip("sLaue CSV データが存在しないためスキップ")

    found_asts = 0
    for csv_p in csv_files:
        meta = get_exposure_time_from_asts(csv_p)
        if meta is not None:
            found_asts += 1
            assert isinstance(meta, ExposureMetadata)
            assert meta.total_seconds in [60.0, 120.0, 180.0]
            assert meta.single_exposure_seconds == 60.0
            assert meta.num_accumulations in [1, 2, 3]

    assert found_asts >= 10, f"少なくとも10個のASTSメタデータが取得できるはずですが、{found_asts}件でした"


def test_exposure_optimization_unsaturated():
    """未飽和データ (184037) の露光解析を検証."""
    p = Path("share/slaue-data/1940/261021_184037_コメント1()_コメント2()_コメント3()_コメント4().CSV")
    if not p.is_file():
        pytest.skip("テスト対象CSVが存在しないためスキップ")

    raw = load_slaue_csv(p)
    res = analyze_exposure_and_optimize(raw, saturation_limit=7.72e6, target_ratio=0.80)

    assert isinstance(res, ExposureOptimizationResult)
    assert res.status == SaturationStatus.UNDER_EXPOSED
    assert res.observed_max < 2.0e6
    assert res.estimated_true_max == res.observed_max
    assert res.recommended_ratio > 1.0  # 延長推奨
    assert res.current_seconds == 120.0
    assert res.recommended_seconds is not None
    assert res.recommended_seconds > res.current_seconds


def test_exposure_optimization_saturated():
    """飽和データ (040502) の露光解析と外挿を検証."""
    p = Path("share/slaue-data/261011_040502_コメント1()_コメント2()_コメント3()_コメント4().CSV")
    if not p.is_file():
        pytest.skip("テスト対象CSVが存在しないためスキップ")

    raw = load_slaue_csv(p)
    res = analyze_exposure_and_optimize(raw, saturation_limit=7.72e6, target_ratio=0.80)

    assert isinstance(res, ExposureOptimizationResult)
    assert res.status == SaturationStatus.SATURATED
    assert res.observed_max >= 7.7e6
    assert res.estimated_true_max >= res.observed_max
    assert res.recommended_ratio < 1.0  # 短縮推奨
    assert res.current_seconds == 180.0
    assert res.recommended_seconds is not None
    assert res.recommended_seconds < res.current_seconds


def test_exposure_section_formatting():
    """セクション文字列フォーマットの検証."""
    p = Path("share/slaue-data/261011_035832_コメント1()_コメント2()_コメント3()_コメント4().CSV")
    if not p.is_file():
        pytest.skip("テスト対象CSVが存在しないためスキップ")

    raw = load_slaue_csv(p)
    res = analyze_exposure_and_optimize(raw)
    sec_text = res.format_section()

    assert "【⏱️ 露光時間最適化＆ピーク解析】" in sec_text
    assert "飽和検知" in sec_text or "上限到達" in sec_text
    assert "推奨測定時間" in sec_text
    assert "counts" in sec_text


def test_pipeline_exposure_integration():
    """パイプライン内での露光時間最適化統合の検証."""
    p = Path("share/slaue-data/1940/261021_184037_コメント1()_コメント2()_コメント3()_コメント4().CSV")
    if not p.is_file():
        pytest.skip("テスト対象CSVが存在しないためスキップ")

    cfg = PipelineConfig(
        run_alignment=True,
        optimize_exposure=True,
        show_diagnostic=False,
        validate_physics=False,
        export_comparison=False,
        export_contrast_optimized=False,
        save_intermediates=False,
    )
    pipeline = AstroLauePipeline(cfg)

    received_feedback = []

    def on_feedback(align_res, exp_res=None):
        received_feedback.append((align_res, exp_res))

    res = pipeline.run(image_source=p, on_alignment_ready=on_feedback)

    assert res.alignment_result is not None
    assert res.exposure_result is not None
    assert len(received_feedback) == 1
    fast_align, fast_exp = received_feedback[0]
    assert fast_align is not None
    assert fast_exp is not None
    assert fast_exp.status == SaturationStatus.UNDER_EXPOSED


def test_exposure_optimization_synthetic():
    """実ファイル非依存の合成データ (未飽和 & 飽和クリッピング) による単体テスト."""
    sat_limit = 7.72e6
    bg = 500000.0

    # 1. 未飽和の合成ガウシアン (ピーク 3,000,000 counts)
    r_grid, th_grid = np.meshgrid(np.linspace(5, 40, 100), np.linspace(0, 360, 200), indexing="ij")
    synth_under = bg + 2500000.0 * np.exp(-((r_grid - 20.0)**2 / (2 * 1.5**2) + (th_grid - 180.0)**2 / (2 * 5.0**2)))
    res_under = analyze_exposure_and_optimize(synth_under, saturation_limit=sat_limit)
    assert res_under.status == SaturationStatus.UNDER_EXPOSED
    assert res_under.recommended_ratio > 1.0

    # 2. 飽和クリップされた合成ガウシアン (真のピーク 10,000,000 counts を 7.72e6 でクリップ)
    true_peak = 10000000.0
    synth_sat = bg + (true_peak - bg) * np.exp(-((r_grid - 20.0)**2 / (2 * 1.5**2) + (th_grid - 180.0)**2 / (2 * 5.0**2)))
    synth_clipped = np.minimum(synth_sat, sat_limit)
    res_sat = analyze_exposure_and_optimize(synth_clipped, saturation_limit=sat_limit)
    assert res_sat.status == SaturationStatus.SATURATED
    assert res_sat.estimated_true_max >= sat_limit
    assert res_sat.recommended_ratio < 1.0
