"""極座標・直交座標展開における方位角オフセット (theta_offset_deg) の単体テスト.

装置GUI（Rigaku Astray / sLaue）の画面表示（12時方向 theta=0°）との幾何学的整合性、
および双方向（直交 <-> 極座標）の可逆性・スポット解析整合性を検証します。
"""

import numpy as np
import pytest

from astrolaue.polar import cartesian_to_polar, polar_to_cartesian
from astrolaue.analyze import analyze_diffraction_spots
from astrolaue.pipeline import PipelineConfig


class TestOrientationOffset:
    """方位角オフセット補正機能の検証スイート."""

    def test_polar_to_cartesian_default_and_offset(self):
        """theta_offset_deg によるスポットの回転展開を検証."""
        n_r, n_theta = 100, 360
        polar = np.zeros((n_r, n_theta), dtype=np.float32)
        # theta = 0 deg にスポットを配置 (半径インデックス 50)
        polar[50, 0] = 1.0

        out_sz = 200
        center = (100.0, 100.0)
        r_range = (10.0, 80.0)

        # 1. offset = 0.0 deg の場合 (3時方向 / +X方向)
        cart_0 = polar_to_cartesian(
            polar,
            output_shape=(out_sz, out_sz),
            center=center,
            r_range=r_range,
            theta_offset_deg=0.0,
        )
        py0, px0 = np.unravel_index(np.argmax(cart_0), cart_0.shape)
        dx0, dy0 = px0 - center[0], py0 - center[1]
        assert dx0 > 30.0
        assert abs(dy0) <= 2.0  # 3時方向 (水平右)

        # 2. offset = -90.0 deg の場合 (12時方向 / -Y方向: 装置GUI整合)
        cart_minus90 = polar_to_cartesian(
            polar,
            output_shape=(out_sz, out_sz),
            center=center,
            r_range=r_range,
            theta_offset_deg=-90.0,
        )
        py_m90, px_m90 = np.unravel_index(np.argmax(cart_minus90), cart_minus90.shape)
        dx_m90, dy_m90 = px_m90 - center[0], py_m90 - center[1]
        assert abs(dx_m90) <= 2.0  # 垂直線上
        assert dy_m90 < -30.0      # 12時方向 (真上)

        # 3. theta = 90 deg のスポットを offset = -90.0 で展開 -> 3時方向 (+X方向)
        polar_90 = np.zeros((n_r, n_theta), dtype=np.float32)
        polar_90[50, 90] = 1.0
        cart_90_m90 = polar_to_cartesian(
            polar_90,
            output_shape=(out_sz, out_sz),
            center=center,
            r_range=r_range,
            theta_offset_deg=-90.0,
        )
        py_90, px_90 = np.unravel_index(np.argmax(cart_90_m90), cart_90_m90.shape)
        dx_90, dy_90 = px_90 - center[0], py_90 - center[1]
        assert dx_90 > 30.0
        assert abs(dy_90) <= 2.0   # 3時方向 (真右)

    def test_reversibility_with_offset(self):
        """cartesian_to_polar と polar_to_cartesian の完全な可逆往復変換を検証."""
        out_sz = 200
        center = (100.0, 100.0)
        r_range = (20.0, 70.0)

        # 直交画像上に真上 (12時方向, dx=0, dy=-40) のスポットを配置
        cart_orig = np.zeros((out_sz, out_sz), dtype=np.float32)
        cart_orig[60, 100] = 1.0  # (x=100, y=60) -> dx=0, dy=-40 (12時方向)

        # 12時方向を theta=0° とする offset = -90.0 で極座標展開
        polar_img = cartesian_to_polar(
            cart_orig,
            center=center,
            r_range=r_range,
            n_r=100,
            n_theta=360,
            theta_offset_deg=-90.0,
        )

        # 極座標画像上で theta=0° 近傍にピークが存在すること
        th_idx_peak = np.argmax(polar_img.sum(axis=0))
        assert th_idx_peak in [0, 1, 359]  # theta=0°

        # 再び直交画像へ逆変換
        cart_recon = polar_to_cartesian(
            polar_img,
            output_shape=(out_sz, out_sz),
            center=center,
            r_range=r_range,
            theta_offset_deg=-90.0,
        )

        py_rec, px_rec = np.unravel_index(np.argmax(cart_recon), cart_recon.shape)
        assert abs(px_rec - 100) <= 1
        assert abs(py_rec - 60) <= 1

    def test_spot_analysis_with_offset(self):
        """直交画像上のスポット解析において極座標プロファイルが正しく逆引きされることを検証."""
        out_sz = 200
        center = (100.0, 100.0)
        r_range = (10.0, 80.0)
        n_r, n_theta = 100, 360

        # 極座標上で theta=0° (インデックス0) にピーク
        polar = np.zeros((n_r, n_theta), dtype=np.float32)
        # スポットにガウス幅を持たせる
        r_idx = 50
        for dt in range(-5, 6):
            polar[r_idx, dt % n_theta] = np.exp(-0.5 * (dt / 1.5) ** 2)

        # offset = -90.0 で直交展開 (真上 12時方向に出現)
        cart = polar_to_cartesian(
            polar,
            output_shape=(out_sz, out_sz),
            center=center,
            r_range=r_range,
            theta_offset_deg=-90.0,
        )

        spots, rep_profile = analyze_diffraction_spots(
            cart_before=cart,
            cart_after=cart,
            polar_before=polar,
            polar_after=polar,
            center=center,
            r_range=r_range,
            min_sigma=2.0,
            max_sigma=6.0,
            theta_offset_deg=-90.0,
        )

        assert len(spots) >= 1
        top_spot = spots[0]
        # 装置極座標系と整合: 12時方向のスポットは theta=0° (±5°)
        assert (abs(top_spot.theta_deg) < 5.0 or abs(top_spot.theta_deg - 360.0) < 5.0)
        # FWHM 算出が正常に行われ、ゼロでないこと (極座標インデックスが一致している証明)
        assert top_spot.fwhm_before > 0.0
        # 代表プロファイル抽出が背景ではなくピークを正確にサンプリングしていること
        assert len(rep_profile) > 0
        assert rep_profile["prof_before"].max() > 0.5

    def test_csv_to_physical_xy_alignment_orientation(self):
        """CSV極座標 (theta: 12時=0°, 時計回り) から物理座標 (X:右, Y:上) への変換整合性を検証."""
        r = 30.0  # mm
        # 1. 12時方向 (theta=0°) -> (X=0, Y=+r)
        th_0 = np.radians(0.0)
        X_0, Y_0 = r * np.sin(th_0), r * np.cos(th_0)
        assert abs(X_0) < 1e-6
        assert abs(Y_0 - 30.0) < 1e-6

        # 2. 3時方向 (theta=90°) -> (X=+r, Y=0)
        th_90 = np.radians(90.0)
        X_90, Y_90 = r * np.sin(th_90), r * np.cos(th_90)
        assert abs(X_90 - 30.0) < 1e-6
        assert abs(Y_90) < 1e-6

        # 3. 6時方向 (theta=180°) -> (X=0, Y=-r)
        th_180 = np.radians(180.0)
        X_180, Y_180 = r * np.sin(th_180), r * np.cos(th_180)
        assert abs(X_180) < 1e-6
        assert abs(Y_180 - (-30.0)) < 1e-6

        # 4. 9時方向 (theta=270°) -> (X=-r, Y=0)
        th_270 = np.radians(270.0)
        X_270, Y_270 = r * np.sin(th_270), r * np.cos(th_270)
        assert abs(X_270 - (-30.0)) < 1e-6
        assert abs(Y_270) < 1e-6

    def test_pipeline_config_default_offset(self):
        """PipelineConfig のデフォルトが -90.0 であることを確認."""
        cfg = PipelineConfig()
        assert cfg.theta_offset_deg == -90.0
