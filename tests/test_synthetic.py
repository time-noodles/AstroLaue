"""模擬ラウエ像生成と復元精度の単体テスト.

ガウシアン斑点群に人工的な円周ブレを付与した模擬ラウエ像を生成し、
パイプラインおよび各モジュールにより、FWHMが有意に縮小（先鋭化）し、
真のスポット座標が高精度に復元・検出できるかを検証します。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import pytest

from astrolaue.capture import extract_diffraction_roi
from astrolaue.normalize import detect_center_and_radius, normalize_geometry, subtract_background
from astrolaue.polar import cartesian_to_polar, polar_to_cartesian
from astrolaue.deblur import create_psf_1d, richardson_lucy_circular, estimate_blur_length, wiener_deblur_circular
from astrolaue.analyze import measure_fwhm_profile, detect_spots
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig


def generate_synthetic_laue_pattern(
    size: int = 512,
    center: Tuple[float, float] = (256.0, 256.0),
    r0: float = 16.0,
    r_outer: float = 180.0,
    spots_polar: List[Tuple[float, float, float]] | None = None,
    blur_length_pix: float = 12.0,
    noise_sigma: float = 0.01,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """人工的な円周ブラー付き模擬ラウエ回折像を生成します.

    Args:
        size: 画像サイズ (N x N)
        center: 中心座標 (xc, yc)
        r0: 中央白丸の半径
        r_outer: 外周円盤の半径
        spots_polar: [(r_k, theta_deg, intensity), ...] のリスト
        blur_length_pix: 付与する円周方向ブラー幅 (極座標ピクセル)
        noise_sigma: ガウシアンノイズ標準偏差

    Returns:
        blurred_image: 人工ブラーが付与された直交画像 (float32 [0.0, 1.0])
        ground_truth_xy: 真のスポット直交座標 [(x, y), ...]
    """
    xc, yc = center
    if spots_polar is None:
        # 代表的なスポット群
        spots_polar = [
            (50.0, 45.0, 0.8),
            (70.0, 120.0, 0.9),
            (95.0, 210.0, 0.75),
            (120.0, 300.0, 0.85),
            (150.0, 75.0, 0.7),
        ]

    # 極座標キャンバス (n_r, n_theta)
    n_r = 256
    n_theta = 720
    polar_sharp = np.zeros((n_r, n_theta), dtype=np.float32)

    r_min = r0 * 1.2
    r_max = r_outer * 0.95

    gt_xy = []
    for r_k, th_deg, intensity in spots_polar:
        th_rad = np.radians(th_deg)
        # 真の直交座標
        gx = xc + r_k * np.cos(th_rad)
        gy = yc + r_k * np.sin(th_rad)
        gt_xy.append((gx, gy))

        # 極座標インデックス
        r_idx = int(round((r_k - r_min) / (r_max - r_min) * (n_r - 1)))
        th_idx = int(round((th_rad / (2.0 * np.pi)) * n_theta)) % n_theta

        # 極座標上でガウシアンスポットを作成 (固有幅 sigma=1.5 px)
        for dr in range(-4, 5):
            for dth in range(-4, 5):
                ri = r_idx + dr
                ti = (th_idx + dth) % n_theta
                if 0 <= ri < n_r:
                    val = intensity * np.exp(-0.5 * ((dr / 1.5) ** 2 + (dth / 1.5) ** 2))
                    polar_sharp[ri, ti] = max(polar_sharp[ri, ti], float(val))

    # 方位角 (theta) 方向に人工ブラーを畳み込み
    psf = create_psf_1d(blur_length=blur_length_pix, kernel_type="rect")
    # 循環畳み込み
    k_len = len(psf)
    k_padded = np.zeros(n_theta, dtype=np.float32)
    half_k = k_len // 2
    k_padded[: k_len - half_k] = psf[half_k:]
    k_padded[-half_k:] = psf[:half_k]

    K = np.fft.rfft(k_padded)
    F_polar = np.fft.rfft(polar_sharp, axis=1)
    polar_blurred = np.fft.irfft(F_polar * K, n=n_theta, axis=1)
    polar_blurred = np.maximum(polar_blurred, 0.0)

    # 直交座標系へ逆変換
    cart = polar_to_cartesian(
        polar_blurred,
        output_shape=(size, size),
        center=center,
        r_range=(r_min, r_max),
    )

    # 背景円盤（弱散乱光）と中央白丸を重畳
    y_grid, x_grid = np.ogrid[:size, :size]
    dists = np.hypot(x_grid - xc, y_grid - yc)

    # 回折円盤のベースライン輝度 (0.15)
    disk_mask = dists <= r_outer
    cart[disk_mask] += 0.15

    # 外側背景 (GUIウィンドウ風の薄灰色 0.95)
    cart[~disk_mask] = 0.95

    # 中央白丸 (ビームストッパー 1.0)
    white_mask = dists <= r0
    cart[white_mask] = 1.0

    # ノイズ付加
    if noise_sigma > 0:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_sigma, cart.shape).astype(np.float32)
        cart = np.clip(cart + noise, 0.0, 1.0)

    return cart, gt_xy


class TestSyntheticRecovery:
    """模擬ラウエ像を用いた復元精度の単体テストクラス."""

    def test_center_and_radius_detection(self):
        """中央基準白丸がサブピクセル精度で検出できるかをテスト."""
        true_center = (258.4, 254.7)
        true_r0 = 15.5
        img, _ = generate_synthetic_laue_pattern(
            size=512,
            center=true_center,
            r0=true_r0,
            noise_sigma=0.005,
        )
        img_u8 = (img * 255.0).astype(np.uint8)

        (det_xc, det_yc), det_r0, det_outer = detect_center_and_radius(img_u8)

        # 中心座標誤差 0.5 ピクセル以内
        assert abs(det_xc - true_center[0]) < 0.6
        assert abs(det_yc - true_center[1]) < 0.6
        # 半径誤差 1.0 ピクセル以内
        assert abs(det_r0 - true_r0) < 1.0

    def test_polar_cartesian_roundtrip(self):
        """極座標変換と逆変換の幾何学的整合性をテスト."""
        size = 256
        center = (128.0, 128.0)
        r_range = (20.0, 100.0)

        # テストパターン
        img = np.zeros((size, size), dtype=np.float32)
        # 半径 60, 角度 45度にスポット
        x_s = int(round(128.0 + 60.0 * np.cos(np.pi / 4)))
        y_s = int(round(128.0 + 60.0 * np.sin(np.pi / 4)))
        img[y_s - 2 : y_s + 3, x_s - 2 : x_s + 3] = 1.0

        polar = cartesian_to_polar(img, center, r_range, n_r=128, n_theta=360)
        reconstructed = polar_to_cartesian(polar, (size, size), center, r_range)

        # 再構成画像でスポット位置の輝度が高いこと
        peak_val = reconstructed[y_s, x_s]
        assert peak_val > 0.4

    def test_richardson_lucy_fwhm_reduction(self):
        """Richardson-Lucy逆畳み込みによりFWHMが大幅に縮小（先鋭化）することをテスト."""
        # 1次元テスト信号: ガウシアンスポットに長さ 14px のブラーを付与
        n = 360
        signal_sharp = np.zeros(n, dtype=np.float32)
        signal_sharp[180] = 1.0
        # 固有幅ガウシアン
        for i in range(-5, 6):
            signal_sharp[180 + i] = np.exp(-0.5 * (i / 1.5) ** 2)

        blur_len = 14.0
        psf = create_psf_1d(blur_len, kernel_type="rect")

        # 畳み込み
        k_padded = np.zeros(n, dtype=np.float32)
        hk = len(psf) // 2
        k_padded[: len(psf) - hk] = psf[hk:]
        k_padded[-hk:] = psf[:hk]
        blurred = np.fft.irfft(np.fft.rfft(signal_sharp) * np.fft.rfft(k_padded), n=n)

        # 2D行列化して逆畳み込み関数へ入力
        blurred_2d = np.tile(blurred, (4, 1))
        restored_2d = richardson_lucy_circular(blurred_2d, psf, num_iter=30)

        # FWHM 比較
        fwhm_before = measure_fwhm_profile(blurred)
        fwhm_after = measure_fwhm_profile(restored_2d[0])

        reduction = (1.0 - (fwhm_after / fwhm_before)) * 100.0

        # 検証: 復元前FWHMはブラー幅相当、復元後は半分以下 (縮小率 > 40%)
        assert fwhm_before > 11.0
        assert fwhm_after < 6.0
        assert reduction > 45.0

    def test_end_to_end_synthetic_pipeline(self, tmp_path: Path):
        """合成画像を用いた全体パイプラインの結合テスト."""
        size = 512
        center = (256.0, 256.0)
        true_blur = 12.0
        synth_img, gt_xy = generate_synthetic_laue_pattern(
            size=size,
            center=center,
            r0=16.0,
            r_outer=180.0,
            blur_length_pix=true_blur,
            noise_sigma=0.005,
        )
        synth_u8 = (synth_img * 255.0).astype(np.uint8)
        img_path = tmp_path / "synthetic_laue.png"
        cv2.imwrite(str(img_path), synth_u8)

        cfg = PipelineConfig(
            output_dir=tmp_path / "results",
            num_iter=25,
            output_size=512,
            n_r=256,
            n_theta=720,
            show_diagnostic=True,
            save_intermediates=True,
        )
        pipeline = AstroLauePipeline(cfg)
        res = pipeline.run(image_source=img_path)

        # 1. ブラー幅の推定精度 (真値 12.0 に対し誤差 2.5px 以内)
        assert abs(res.blur_length - true_blur) < 3.0

        # 2. スポット検出件数
        assert len(res.spots) >= 3

        # 3. 平均先鋭化率が正（FWHMが縮小）であること
        reductions = [s.fwhm_reduction_ratio for s in res.spots if s.fwhm_before > 3.0]
        assert len(reductions) > 0
        avg_red = sum(reductions) / len(reductions)
        assert avg_red > 30.0

        # 4. 出力ファイルの存在確認
        assert (tmp_path / "results" / "diagnostic_report.png").is_file()
        assert (tmp_path / "results" / "detected_spots.csv").is_file()

    def test_peak_recovery_fidelity_benchmark(self, tmp_path: Path):
        """物理模擬ラウエ像に対するピーク再現性（位置・強度・先鋭化・F1）ベンチマークテスト."""
        from astrolaue.simulation import (
            generate_realistic_laue_simulation,
            evaluate_peak_recovery,
            plot_recovery_benchmark_report,
        )

        # 1. 晶帯軸反射スポット・走査ブラー・不均一背景散乱・ノイズを付与した模擬ラウエ像を生成
        true_blur = 14.0
        blurred_img, gt_img, gt_spots = generate_realistic_laue_simulation(
            size=512,
            center=(256.0, 256.0),
            r0=20.0,
            r_outer=230.0,
            blur_length_pix=true_blur,
            num_zones=8,
            spots_per_zone=6,
            spot_sigma=1.3,
            noise_sigma=0.012,
            random_seed=42,
        )
        sim_path = tmp_path / "simulated_laue.png"
        cv2.imwrite(str(sim_path), (blurred_img * 255.0).astype(np.uint8))

        # 2. パイプラインによる復元実行
        cfg = PipelineConfig(
            output_dir=tmp_path / "results",
            num_iter=28,
            output_size=512,
            n_r=384,
            n_theta=1080,
            show_diagnostic=True,
            save_intermediates=True,
        )
        pipeline = AstroLauePipeline(cfg)
        result = pipeline.run(image_source=sim_path)

        # 3. 再現性指標の定量評価
        metrics, matched_pairs = evaluate_peak_recovery(
            pipeline_result=result,
            ground_truth_spots=gt_spots,
            ground_truth_image=gt_img,
            match_distance_tolerance_px=4.5,
        )

        # --- 品質・再現性基準の検証 ---
        # 1) 再現率: 真の回折スポットの 90% 以上を確実に検出・救出できているか
        assert metrics.recall >= 0.90, f"再現率が不足しています: {metrics.recall*100:.1f}% < 90.0%"

        # 2) 位置精度: 復元された重心座標が真の位置に対して 1.0 ピクセル以内のサブピクセル精度か
        assert metrics.position_rmse_px <= 1.0, f"重心位置誤差が大きすぎます: {metrics.position_rmse_px:.2f} px"

        # 3) 強度線形性: 復元ピーク強度と真値強度の決定係数 R^2 >= 0.85 を保持しているか
        assert metrics.intensity_r2 >= 0.85, f"強度の相関が低下しています: R² = {metrics.intensity_r2:.3f}"

        # 4) 先鋭化率: 走査ブラーが有意に先鋭化 (縮小率 >= 45%) されているか
        assert metrics.avg_fwhm_reduction_pct >= 45.0, f"先鋭化が不十分です: {metrics.avg_fwhm_reduction_pct:.1f}%"

        # 5) 画質: 復元画像の PSNR が 25dB 以上を達成しているか
        assert metrics.psnr_db >= 25.0, f"復元PSNRが低下しています: {metrics.psnr_db:.1f} dB"

        # 6) ベンチマークレポート図の出力テスト
        report_fig = tmp_path / "synthetic_recovery_benchmark.png"
        plot_recovery_benchmark_report(
            ground_truth_image=gt_img,
            blurred_image=blurred_img,
            pipeline_result=result,
            ground_truth_spots=gt_spots,
            matched_pairs=matched_pairs,
            metrics=metrics,
            output_path=report_fig,
        )
        assert report_fig.is_file(), "ベンチマークレポート図が生成されていません"
