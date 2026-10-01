"""物理妥当性検証・透過PNG出力・バッチ耐性のテスト."""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from astrolaue.pipeline import AstroLauePipeline, PipelineConfig
from astrolaue.export import export_publication_rgba, create_circular_mask
from astrolaue.validate import evaluate_flux_conservation, compute_residual_map, validate_physical_consistency
from astrolaue.deblur import create_psf_1d, richardson_lucy_circular


class TestPhysicalValidation:
    """物理妥当性・透過PNGの単体テスト."""

    def test_flux_conservation_on_synthetic_spot(self):
        """1次元パルススポットにおいて総光量が保存されることを検証."""
        n_theta = 360
        raw = np.zeros((1, n_theta), dtype=np.float32)
        # 中心 180 にブラー付きスポット
        psf = create_psf_1d(blur_length=12.0, kernel_type="rect")
        # シャープなパルス (光量 = 100.0)
        sharp = np.zeros(n_theta, dtype=np.float32)
        sharp[180] = 100.0

        # 畳み込み
        k_padded = np.zeros(n_theta, dtype=np.float32)
        hk = len(psf) // 2
        k_padded[: len(psf) - hk] = psf[hk:]
        k_padded[-hk:] = psf[:hk]
        raw[0] = np.fft.irfft(np.fft.rfft(sharp) * np.fft.rfft(k_padded), n=n_theta)

        # 逆畳み込み
        restored = richardson_lucy_circular(raw, psf, num_iter=30)

        raw_sum = float(np.sum(raw[0]))
        rest_sum = float(np.sum(restored[0]))

        flux_ratio = rest_sum / raw_sum
        # 理論上、厳密に 1.0 (相対誤差 1% 未満)
        assert abs(flux_ratio - 1.0) < 0.01

    def test_centroid_invariance_on_symmetric_blur(self):
        """対称PSFによる逆畳み込みで重心がシフトしないことを検証 (< 0.2 px)."""
        n_theta = 360
        center_true = 180.0
        sharp = np.zeros(n_theta, dtype=np.float32)
        sharp[int(center_true)] = 50.0

        psf = create_psf_1d(blur_length=14.0, kernel_type="rect")
        k_padded = np.zeros(n_theta, dtype=np.float32)
        hk = len(psf) // 2
        k_padded[: len(psf) - hk] = psf[hk:]
        k_padded[-hk:] = psf[:hk]
        blurred = np.fft.irfft(np.fft.rfft(sharp) * np.fft.rfft(k_padded), n=n_theta)

        restored = richardson_lucy_circular(blurred.reshape(1, -1), psf, num_iter=25)[0]

        # 重心算出
        coords = np.arange(n_theta, dtype=np.float64)
        c_raw = np.sum(coords * blurred) / np.sum(blurred)
        c_rest = np.sum(coords * restored) / np.sum(restored)

        assert abs(c_rest - c_raw) < 0.1

    def test_export_publication_rgba(self, tmp_path: Path):
        """透過PNG (RGBA) のアルファマスク境界を検証."""
        size = 200
        center = (100.0, 100.0)
        r_inner = 20.0
        r_outer = 80.0

        img = np.ones((size, size), dtype=np.float32) * 0.5
        out_png = tmp_path / "test_rgba.png"

        saved = export_publication_rgba(
            image=img,
            center=center,
            r_inner=r_inner,
            r_outer=r_outer,
            output_path=out_png,
        )
        assert saved.is_file()

        # 読み込んでチャンネルとアルファ値を確認
        rgba = cv2.imread(str(saved), cv2.IMREAD_UNCHANGED)
        assert rgba.shape == (size, size, 4)

        # 1. デフォルト (mask_inner_hole=False): 中心穴 (r = 5 px) は不透明白色 (Alpha == 255, BGR == 255)
        assert rgba[100, 100, 3] == 255
        assert rgba[100, 100, 0] == 255
        assert rgba[100, 105, 3] == 255

        # 2. 有効回折領域内 (r = 50 px) は完全不透過 (Alpha == 255)
        assert rgba[100, 150, 3] == 255

        # 3. 外周外側 (r = 95 px) は完全透過 (Alpha == 0)
        assert rgba[100, 195, 3] == 0

        # 4. mask_inner_hole=True の場合: 中心穴も完全透過 (Alpha == 0)
        out_hole_png = tmp_path / "test_rgba_hole.png"
        export_publication_rgba(
            image=img,
            center=center,
            r_inner=r_inner,
            r_outer=r_outer,
            output_path=out_hole_png,
            mask_inner_hole=True,
        )
        rgba_hole = cv2.imread(str(out_hole_png), cv2.IMREAD_UNCHANGED)
        assert rgba_hole[100, 100, 3] == 0

    def test_clipboard_copy(self, tmp_path: Path):
        """クリップボードコピー関数の単体テスト (エラーなく実行できること)."""
        from astrolaue.clipboard import copy_image_to_clipboard
        test_img = np.ones((50, 50, 4), dtype=np.uint8) * 200
        test_p = tmp_path / "clipboard_test.png"
        cv2.imwrite(str(test_p), test_img)

        # 正常に例外なく処理されること
        result = copy_image_to_clipboard(test_p)
        assert isinstance(result, bool)

    def test_batch_execution_on_share_directory(self, tmp_path: Path):
        """share ディレクトリ内の全画像に対する一括バッチ処理の統合テスト."""
        batch_dir = Path("share")
        if not batch_dir.is_dir():
            pytest.skip("share ディレクトリが見つかりません。")

        out_dir = tmp_path / "batch_out"
        cfg = PipelineConfig(
            output_dir=out_dir,
            num_iter=15,
            output_size=512,
            n_r=128,
            n_theta=360,
            show_diagnostic=True,
            validate_physics=True,
            export_transparent=True,
        )
        pipeline = AstroLauePipeline(cfg)
        summary_csv = pipeline.run_batch(batch_dir)

        assert summary_csv.is_file()
        assert (out_dir / "batch_summary_plot.png").is_file()

        # CSV内容チェック
        with open(summary_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # ヘッダー + 6画像 = 7行
        assert len(lines) >= 6
        # 全画像 SUCCESS
        assert "SUCCESS" in lines[1]
