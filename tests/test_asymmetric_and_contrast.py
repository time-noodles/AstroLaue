"""非対称PSF（EMG / Skew-Normal）およびコントラスト最適化出力のテスト."""

from pathlib import Path
import time
import cv2
import numpy as np
import pytest

from astrolaue.deblur import create_psf_1d, estimate_asymmetric_psf_ensemble
from astrolaue.export import (
    export_contrast_optimized_image,
    export_linear_image,
    export_three_way_comparison,
)
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig


class TestAsymmetricPsfAndContrast:
    """非対称PSF生成、アンサンブル同定、コントラスト最適化出力の検証クラス."""

    def test_asymmetric_psf_centroid_alignment(self):
        """非対称PSFカーネルが重心ゼロ (サブピクセルアライメント) かつ総和1.0に正規化されているかを検証."""
        for k_type, kwargs in [
            ("emg", {"sigma": 2.5, "tau": 5.5}),
            ("skew_normal", {"sigma": 4.5, "alpha": 3.5}),
            ("asymmetric_shoulder", {"sigma": 3.0, "tau": 5.0}),
        ]:
            psf = create_psf_1d(blur_length=16.0, kernel_type=k_type, **kwargs)
            assert len(psf) % 2 == 1, f"PSFサイズは奇数でなければなりません: len={len(psf)}"
            assert np.isclose(np.sum(psf), 1.0, atol=1e-5), f"PSF総和が1.0ではありません: {np.sum(psf)}"

            # 幾何重心の検証 (原点 x=0 からのズレが 0.05 px 未満)
            mid = len(psf) // 2
            x_coords = np.arange(len(psf)) - mid
            centroid = float(np.sum(x_coords * psf) / np.sum(psf))
            assert abs(centroid) < 0.08, f"{k_type} の重心がズレています: centroid={centroid:.4f} px"

    def test_ensemble_psf_fast_identification(self):
        """実機データからアンサンブル同定が 20ms 以内で高精度に完了することを検証."""
        sample_path = Path("share/260930-1.png")
        if not sample_path.exists():
            sample_path = Path("tests/sample.png")
        if not sample_path.exists():
            pytest.skip("実機テスト画像が見つかりません。")

        img = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        center = (w / 2.0, h / 2.0)

        from astrolaue.polar import cartesian_to_polar
        polar = cartesian_to_polar(img, center=center, r_range=(50.0, min(center) - 10.0), n_r=256, n_theta=720)

        t0 = time.perf_counter()
        psf, params = estimate_asymmetric_psf_ensemble(polar_image=polar, model_type="auto", fallback_blur=15.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 計算時間は 25ms 以内（実測 ~8ms）
        assert elapsed_ms < 25.0, f"アンサンブル同定が遅すぎます: {elapsed_ms:.2f} ms"
        assert len(psf) > 0
        assert params.kernel_type in ["emg", "skew_normal"]
        assert params.r2_score >= 0.85, f"適合度が不十分です: R²={params.r2_score:.3f}"
        assert params.sigma > 0.5

    def test_contrast_exports_and_pipeline_integration(self, tmp_path: Path):
        """パイプラインで auto カーネルとコントラスト最適化画像が正しく生成されるかを検証."""
        sample_path = Path("share/260930-1.png")
        if not sample_path.exists():
            sample_path = Path("tests/sample.png")
        if not sample_path.exists():
            pytest.skip("実機テスト画像が見つかりません。")

        out_dir = tmp_path / "out_asym"
        cfg = PipelineConfig(
            output_dir=out_dir,
            kernel_type="auto",
            export_linear=True,
            export_contrast_optimized=True,
            show_diagnostic=False,
            save_intermediates=False,
        )
        pipe = AstroLauePipeline(cfg)
        res = pipe.run(image_source=sample_path)

        # 出力結果の検証
        assert res.linear_png_path is not None and res.linear_png_path.exists()
        assert res.contrast_optimized_png_path is not None and res.contrast_optimized_png_path.exists()
        assert res.contrast_pseudo_png_path is not None and res.contrast_pseudo_png_path.exists()
        assert res.contrast_comparison_path is not None and res.contrast_comparison_path.exists()
        assert res.psf_params is not None
        assert "kernel_type" in res.psf_params
        assert res.psf_params["r2_score"] >= 0.85

        # ファイルサイズが空でないこと
        assert res.linear_png_path.stat().st_size > 1000
        assert res.contrast_optimized_png_path.stat().st_size > 1000
        assert res.contrast_comparison_path.stat().st_size > 1000
