"""背面反射ラウエ結晶主軸自動同定 & ゴニオメーター補正角推定モジュールの単体テスト."""

import pytest
import numpy as np
import cv2
from pathlib import Path

from astrolaue.alignment import (
    LaueAlignmentConfig,
    detector_to_normals,
    normals_to_gnomonic,
    gnomonic_to_detector,
    zone_line_to_detector_hyperbola,
    calculate_goniometer_angles,
    detect_zone_lines,
    find_principal_axis_candidates,
    solve_laue_alignment,
    generate_synthetic_laue_sample,
    extract_laue_spots_from_image,
    plot_alignment_diagnostics,
)


class TestLaueAlignmentGeometry:
    """幾何学計算および可逆変換の数学的完全性のテスト."""

    def test_center_spot_normal(self):
        """検出器中心 (0, 0) のスポットは法線 (0, 0, 1) およびグノモン (0, 0) に対応することを検証."""
        L = 40.0
        center_xy = np.array([[0.0, 0.0]], dtype=np.float64)
        normals = detector_to_normals(center_xy, camera_length_mm=L)
        assert normals.shape == (1, 3)
        np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-7)

        uv = normals_to_gnomonic(normals)
        assert uv.shape == (1, 2)
        np.testing.assert_allclose(uv[0], [0.0, 0.0], atol=1e-7)

    def test_roundtrip_detector_to_gnomonic_to_detector(self):
        """検出器座標 -> 法線 -> グノモン座標 -> 検出器座標 の厳密な可逆性を検証."""
        L = 50.0
        test_pts = np.array([
            [0.0, 0.0],
            [12.5, -8.3],
            [-15.0, 20.0],
            [5.0, 30.0],
            [-22.0, -18.0],
        ], dtype=np.float64)

        normals = detector_to_normals(test_pts, camera_length_mm=L)
        uv = normals_to_gnomonic(normals)
        reconstructed_pts = gnomonic_to_detector(uv, camera_length_mm=L)

        np.testing.assert_allclose(reconstructed_pts, test_pts, atol=1e-6)

    def test_hyperbola_generation(self):
        """晶帯直線から検出器上の双曲線サンプリング点列が生成できることを検証."""
        from astrolaue.alignment import ZoneLine
        L = 40.0
        # a*u + b*v + c = 0 (傾いた直線)
        dummy_zone = ZoneLine(
            line_id=1, a=0.7071, b=-0.7071, c=-0.05,
            score=10.0, num_inliers=10, inlier_indices=list(range(10)),
            zone_axis_norm=(0.7071, -0.7071, -0.05),
        )
        hyp = zone_line_to_detector_hyperbola(dummy_zone, camera_length_mm=L, num_points=100)
        assert len(hyp) > 10
        assert hyp.shape[1] == 2
        # 双曲線上にあることを確認
        assert not np.any(np.isnan(hyp))


class TestGoniometerCalculation:
    """ゴニオメーター補正回転角計算のテスト."""

    def test_zero_tilt_gives_zero_angles(self):
        """傾きゼロの主軸 (u=0, v=0) は補正角 0.0° となることを検証."""
        cfg = LaueAlignmentConfig(stage_order="XY")
        norm, thx, thy, tilt = calculate_goniometer_angles(0.0, 0.0, config=cfg)
        np.testing.assert_allclose(norm, [0.0, 0.0, 1.0], atol=1e-7)
        assert abs(thx) < 1e-7
        assert abs(thy) < 1e-7
        assert abs(tilt) < 1e-7

    def test_xy_and_yx_stage_orders(self):
        """XY および YX ステージ構成で適切な角度が算出されることを検証."""
        u_target = 0.08
        v_target = -0.05
        cfg_xy = LaueAlignmentConfig(stage_order="XY")
        cfg_yx = LaueAlignmentConfig(stage_order="YX")

        norm_xy, thx_xy, thy_xy, tilt_xy = calculate_goniometer_angles(u_target, v_target, config=cfg_xy)
        norm_yx, thx_yx, thy_yx, tilt_yx = calculate_goniometer_angles(u_target, v_target, config=cfg_yx)

        # 法線ベクトルおよび全傾き角はステージ順序によらず不変
        np.testing.assert_allclose(norm_xy, norm_yx, atol=1e-7)
        assert abs(tilt_xy - tilt_yx) < 1e-7

        # 符号と大きさが合理的
        assert thy_xy < 0.0  # u > 0 なので Y回転はマイナス
        assert thx_xy < 0.0  # v < 0 なので X回転はマイナス


class TestSyntheticAlignmentAccuracy:
    """合成ラウエスポット群を用いたサブ度精度逆算のベンチマークテスト."""

    @pytest.mark.parametrize("true_thx, true_thy, lattice", [
        (6.0, -4.5, "tetragonal"),
        (-3.5, 5.0, "cubic"),
        (4.2, 3.8, "hexagonal"),
    ])
    def test_subdegree_recovery_accuracy(self, true_thx: float, true_thy: float, lattice: str):
        """未知結晶の様々な初期傾き角に対し、0.1°以下のサブ度精度で補正角が逆算できることを検証."""
        L_cam = 40.0
        spots_xy, intensities, meta = generate_synthetic_laue_sample(
            true_tilt_x_deg=true_thx,
            true_tilt_y_deg=true_thy,
            camera_length_mm=L_cam,
            lattice_type=lattice,
            num_fake_spots=10,
            noise_sigma_mm=0.03,
            random_seed=42,
        )

        cfg = LaueAlignmentConfig(
            camera_length_mm=L_cam,
            dist_thresh=0.016,
            min_inliers=4,
            max_zones=6,
            stage_order="XY",
        )

        result = solve_laue_alignment(spots_xy, intensities=intensities, config=cfg)
        best = result.best_candidate

        expected_thx = -true_thx
        expected_thy = -true_thy

        error_x = abs(best.delta_theta_x_deg - expected_thx)
        error_y = abs(best.delta_theta_y_deg - expected_thy)

        assert error_x < 0.10, f"X軸補正角誤差が大きすぎます: {error_x:.3f}° (期待値: {expected_thx:+.2f}°, 推定値: {best.delta_theta_x_deg:+.2f}°)"
        assert error_y < 0.10, f"Y軸補正角誤差が大きすぎます: {error_y:.3f}° (期待値: {expected_thy:+.2f}°, 推定値: {best.delta_theta_y_deg:+.2f}°)"

    def test_fallback_with_very_few_spots(self):
        """晶帯直線が交差しない極小スポット数（3点）の場合にフォールバックが正しく作動することを検証."""
        spots_xy = np.array([
            [5.0, 5.0],
            [12.0, -8.0],
            [-10.0, 15.0],
        ], dtype=np.float64)
        intensities = np.array([0.9, 0.4, 0.2], dtype=np.float64)

        cfg = LaueAlignmentConfig(camera_length_mm=40.0)
        result = solve_laue_alignment(spots_xy, intensities=intensities, config=cfg)

        assert len(result.candidates) >= 1
        best = result.best_candidate
        assert best.rank == 1
        # 最高強度スポット (5.0, 5.0) 付近にフォールバックしていることを確認
        assert abs(best.detector_x_mm - 5.0) < 1e-4
        assert abs(best.detector_y_mm - 5.0) < 1e-4

    def test_diagnostic_plot_generation(self, tmp_path: Path):
        """診断プロットの生成と保存が正常に行われることを検証."""
        spots_xy, intensities, _ = generate_synthetic_laue_sample(
            true_tilt_x_deg=2.0,
            true_tilt_y_deg=-3.0,
            camera_length_mm=40.0,
            num_fake_spots=5,
            random_seed=99,
        )
        cfg = LaueAlignmentConfig(camera_length_mm=40.0)
        result = solve_laue_alignment(spots_xy, intensities=intensities, config=cfg)

        out_img = tmp_path / "test_align_plot.png"
        fig = plot_alignment_diagnostics(result, save_path=out_img, show=False)
        assert fig is not None
        assert out_img.is_file()
        assert out_img.stat().st_size > 10000


class TestImagePeakExtraction:
    """画像からのスポット抽出関数のテスト."""

    def test_extract_spots_from_synthetic_image(self):
        """合成画像からピーク検出および物理座標変換が正しく行われることを検証."""
        h, w = 400, 400
        img = np.zeros((h, w), dtype=np.uint8)

        # 3つのスポットを描画
        cv2.circle(img, (200, 200), 3, 255, -1)  # 中心
        cv2.circle(img, (250, 150), 3, 200, -1)  # 右上 (+X, +Y)
        cv2.circle(img, (150, 250), 3, 180, -1)  # 左下 (-X, -Y)

        cfg = LaueAlignmentConfig(
            pixel_size_mm=0.1,
            beam_center_xy=(200.0, 200.0),
            threshold=0.01,
            tophat_kernel_size=7,
        )

        spots_xy, intensities = extract_laue_spots_from_image(img, config=cfg)
        assert len(spots_xy) == 3

        # 中心のスポット (200, 200) -> (0.0, 0.0) mm
        dists_to_origin = np.hypot(spots_xy[:, 0], spots_xy[:, 1])
        assert np.min(dists_to_origin) < 0.2
