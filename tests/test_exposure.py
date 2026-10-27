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
    assert res.single_exposure_seconds == 60.0
    assert res.num_accumulations == 2
    assert res.recommended_seconds is not None
    assert res.recommended_single_seconds is not None
    assert res.recommended_seconds > res.current_seconds
    assert res.recommended_single_seconds > res.single_exposure_seconds


def test_exposure_optimization_saturated():
    """飽和データ (040502) の露光解析と外挿を検証."""
    p = Path("share/slaue-data/261011_040502_コメント1()_コメント2()_コメント3()_コメント4().CSV")
    if not p.is_file():
        pytest.skip("テスト対象CSVが存在しないためスキップ")

    raw = load_slaue_csv(p)
    res = analyze_exposure_and_optimize(raw, saturation_limit=7.72e6, target_ratio=0.80)

    assert isinstance(res, ExposureOptimizationResult)
    assert res.status == SaturationStatus.SATURATED
    assert res.observed_max >= 7.3e6
    assert res.estimated_true_max >= res.observed_max
    assert res.num_candidate_peaks >= 2
    assert res.radial_integral > 1.0e7
    assert res.recommended_ratio < 1.0  # 短縮推奨
    assert res.current_seconds == 180.0
    assert res.single_exposure_seconds == 60.0
    assert res.num_accumulations == 3
    assert res.recommended_seconds is not None
    assert res.recommended_single_seconds is not None
    assert res.recommended_seconds < res.current_seconds
    assert res.recommended_single_seconds < res.single_exposure_seconds
    assert abs(res.recommended_single_seconds * res.num_accumulations - res.recommended_seconds) < 1.0


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
    assert "測定時間の設定" in sec_text
    assert "counts" in sec_text
    # 現場での誤解を防ぐため、積算回数や総照射時間の文言が含まれていないことを検証
    assert "積算回数" not in sec_text
    assert "総照射時間" not in sec_text


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


def test_exposure_optimization_synthetic_multi_peak_screening():
    """複数飽和ピークが存在する場合に、面積(Radial Integral)最大のピークが正しく選定されるかを実ファイル非依存で検証."""
    sat_limit = 7.72e6
    bg = 500000.0
    r_grid, th_grid = np.meshgrid(np.linspace(5, 40, 150), np.linspace(0, 360, 300), indexing="ij")

    # ピークA: 鋭い飽和ピーク (真の高さ 8.5e6, sigma_r = 1.0, 面積 小)
    peak_a = 8500000.0 * np.exp(-((r_grid - 15.0)**2 / (2 * 1.0**2) + (th_grid - 90.0)**2 / (2 * 4.0**2)))
    # ピークB: 幅広い飽和ピーク (真の高さ 8.5e6, sigma_r = 2.5, 面積 大 -> 支配的ピーク)
    peak_b = 8500000.0 * np.exp(-((r_grid - 30.0)**2 / (2 * 2.5**2) + (th_grid - 270.0)**2 / (2 * 4.0**2)))

    synth_multi = bg + peak_a + peak_b
    synth_clipped = np.minimum(synth_multi, sat_limit)

    res = analyze_exposure_and_optimize(synth_clipped, saturation_limit=sat_limit)

    assert res.status == SaturationStatus.SATURATED
    assert res.num_candidate_peaks >= 2
    # より広い面積を持つ ピークB (r ≒ 30.0 mm) が最強ピークとして選ばれていること
    assert abs(res.peak_r_mm - 30.0) < 1.0
    assert abs(res.peak_theta_deg - 270.0) < 5.0


def test_exposure_section_formatting_synthetic_branches():
    """ASTSメタデータ不在時のフォールバック表示およびSLIGHTLY_HIGHメッセージの検証."""
    sat_limit = 7.72e6
    bg = 500000.0
    r_grid, th_grid = np.meshgrid(np.linspace(5, 40, 50), np.linspace(0, 360, 100), indexing="ij")

    # ASTS不在 (配列入力)
    synth = bg + 2000000.0 * np.exp(-((r_grid - 20.0)**2 / (2 * 1.5**2) + (th_grid - 180.0)**2 / (2 * 5.0**2)))
    res = analyze_exposure_and_optimize(synth, saturation_limit=sat_limit)
    assert res.current_seconds is None
    sec_text = res.format_section()
    assert "変更を推奨" in sec_text
    assert "現行設定の" in sec_text

    # SLIGHTLY_HIGH (飽和寸前: 88〜95%)
    peak_gaussian = np.exp(-((r_grid - 20.0)**2 / (2 * 1.5**2) + (th_grid - 180.0)**2 / (2 * 5.0**2)))
    synth_high = bg + (sat_limit * 0.91 - bg) * (peak_gaussian / peak_gaussian.max())
    res_high = analyze_exposure_and_optimize(synth_high, saturation_limit=sat_limit)
    assert res_high.status == SaturationStatus.SLIGHTLY_HIGH
    sec_text_high = res_high.format_section()
    assert "飽和マージンを確保" in sec_text_high


def test_exposure_section_formatting_long_exposure_advice():
    """推奨露光時間が600秒を超える場合の段階的延長アドバイス出力を検証."""
    sat_limit = 7.72e6
    res = ExposureOptimizationResult(
        status=SaturationStatus.UNDER_EXPOSED,
        observed_max=500000.0,
        estimated_true_max=500000.0,
        background=100000.0,
        saturation_limit=sat_limit,
        target_max=sat_limit * 0.80,
        recommended_ratio=12.35,
        peak_r_mm=20.0,
        peak_theta_deg=180.0,
        current_seconds=120.0,
        single_exposure_seconds=60.0,
        recommended_seconds=1482.0,  # 600秒超
        recommended_single_seconds=741.0,
        num_accumulations=2,
    )
    sec_text = res.format_section()
    assert "推奨時間が10分を超えるため" in sec_text
    assert "段階的な延長" in sec_text
