"""動径背景適応減算およびコントラスト最適化・台形化防止の単体テスト."""

from pathlib import Path
import glob
import numpy as np
import pytest

from astrolaue.polar import subtract_radial_background, polar_to_cartesian
from astrolaue.io import load_slaue_csv, SlaueRawData
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig
from astrolaue.export import export_contrast_optimized_image
import cv2


def test_subtract_radial_background_synthetic():
    """合成極座標データを用いた動径背景減算の平坦化検証."""
    n_r, n_theta = 100, 360
    # 動径方向に単調増加する背景勾配: B(r) = 1000 + 50 * r + ランダムノイズ
    r_idx = np.arange(n_r, dtype=np.float32)[:, None]
    bg_synthetic = 1000.0 + 50.0 * r_idx
    noise = np.random.normal(loc=0.0, scale=30.0, size=(n_r, n_theta)).astype(np.float32)

    # 回折スポットを3箇所に配置
    spots = np.zeros((n_r, n_theta), dtype=np.float32)
    spots[20, 50] = 5000.0  # 内周スポット
    spots[50, 180] = 4000.0 # 中周スポット
    spots[80, 300] = 3000.0 # 外周スポット

    synthetic_data = bg_synthetic + noise + spots

    sub_data, bg_profile = subtract_radial_background(synthetic_data, quantile=0.15, sigma=3.0)

    # 1. 形状チェック
    assert sub_data.shape == (n_r, n_theta)
    assert bg_profile.shape == (n_r,)
    assert np.all(sub_data >= 0.0)

    # 2. 背景プロファイルが動径勾配を正確に捉えているか
    # r=0 付近: ~1000, r=99 付近: ~5950
    assert abs(bg_profile[0] - 1000.0) < 100.0
    assert abs(bg_profile[-1] - 5950.0) < 150.0

    # 3. 減算後の背景中央値が内周・外周ともに均一（ゼロ付近）になっているか
    inner_bg_med = float(np.median(sub_data[10]))
    outer_bg_med = float(np.median(sub_data[85]))
    assert abs(inner_bg_med - outer_bg_med) < 50.0

    # 4. 回折スポットが損なわれず保持されているか
    assert sub_data[20, 50] > 4500.0
    assert sub_data[50, 180] > 3500.0
    assert sub_data[80, 300] > 2500.0


def test_to_normalized_polar_no_trapezoid_plateau():
    """線形スケーリングによりピーク頂上の台形化（フラットトップ）が防止されることを検証."""
    csv_files = sorted(glob.glob("share/**/*.CSV", recursive=True))
    if not csv_files:
        pytest.skip("検証用 CSV データが存在しないためスキップ")

    raw = load_slaue_csv(csv_files[0])

    # 1. デフォルト (linear max, high_percentile=None)
    norm_linear = raw.to_normalized_polar()
    assert norm_linear.max() <= 1.0
    assert norm_linear.min() >= 0.0

    # 最も強いピークの近傍で、強度が 1.0 に張り付いている画素数が 1 のみ（台形フラットトップなし）であることを検証
    max_count = int(np.sum(norm_linear >= 0.9999))
    assert max_count <= 2, f"ピーク頂上が台形化しています: 1.0の画素数={max_count}"

    # 2. 過去の 99.8% クリップ指定時との比較
    norm_clipped = raw.to_normalized_polar(high_percentile=99.8, use_radial_bg=False)
    clipped_count = int(np.sum(norm_clipped >= 0.9999))
    assert clipped_count > 5, "99.8% クリップ時は複数画素が台形化するはず"


def test_contrast_optimized_outer_ring_darkness(tmp_path):
    """動径背景減算により外周リングのコントラストが引き締まり（灰度 <= 10）、白濁しないことを検証."""
    csv_files = sorted(glob.glob("share/1029-report/*.CSV"))
    if not csv_files:
        pytest.skip("1029-report CSV データが存在しないためスキップ")

    raw = load_slaue_csv(csv_files[0])
    norm_polar = raw.to_normalized_polar(use_radial_bg=True)

    out_sz = 512
    norm_center = (out_sz / 2.0, out_sz / 2.0)
    norm_r_outer = (out_sz / 2.0) * 0.95
    norm_r0 = norm_r_outer * (float(raw.radii_mm.min()) / float(raw.radii_mm.max()))
    r_range = (norm_r0, norm_r_outer)

    cart = polar_to_cartesian(norm_polar, output_shape=(out_sz, out_sz), center=norm_center, r_range=r_range)

    out_png = tmp_path / "contrast_test.png"
    export_contrast_optimized_image(
        image=cart,
        center=norm_center,
        r_inner=norm_r0,
        r_outer=norm_r_outer,
        output_path=out_png,
        asinh_beta=6.0,
        export_pseudo_color=False,
    )

    img = cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    h, w = img.shape
    y_idx, x_idx = np.indices((h, w))
    r_map = np.hypot(x_idx - norm_center[0], y_idx - norm_center[1])

    # 最外周リング (r_outer 近傍) の中央値灰度を検証
    outer_mask = (r_map >= norm_r_outer - 20) & (r_map <= norm_r_outer - 5)
    outer_median = float(np.median(img[outer_mask]))
    assert outer_median <= 10.0, f"外周リングが白濁しています: median={outer_median}"


def test_pipeline_integration_with_radial_bg(tmp_path):
    """パイプライン実行時に動径背景減算と線形正規化が正常に統合動作することを検証."""
    csv_files = sorted(glob.glob("share/1029-report/*.CSV"))
    if not csv_files:
        pytest.skip("1029-report CSV データが存在しないためスキップ")

    cfg = PipelineConfig(
        output_dir=tmp_path / "results",
        use_radial_bg=True,
        radial_bg_quantile=0.15,
        num_iter=15,
        show_diagnostic=True,
        save_intermediates=True,
    )
    pipe = AstroLauePipeline(config=cfg)
    res = pipe.run(image_source=Path(csv_files[0]))

    assert res.representative_profile is not None
    rep = res.representative_profile
    assert "prof_before" in rep
    pb = rep["prof_before"]

    # 代表プロファイルのフラットトップ（台形）が存在しないことを検証
    top_plateau_count = int(np.sum(pb >= pb.max() * 0.9999))
    assert top_plateau_count <= 2, f"代表スポットプロファイルが台形化しています: {top_plateau_count}画素"
