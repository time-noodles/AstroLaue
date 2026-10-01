"""全体パイプラインの統合テスト."""

from __future__ import annotations

from pathlib import Path
import pytest

from astrolaue.capture import load_image, extract_diffraction_roi
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig
from astrolaue.main import parse_args, main


class TestPipelineIntegration:
    """実画像および主要コンポーネントの結合テストクラス."""

    @pytest.fixture
    def sample_image_path(self) -> Path | None:
        p = Path("share/画像260925.png")
        if p.is_file():
            return p
        alt = Path("/share/画像260925.png")
        if alt.is_file():
            return alt
        return None

    def test_real_sample_pipeline_execution(self, sample_image_path: Path | None, tmp_path: Path):
        """実サンプル画像を用いたエンドツーエンド処理テスト."""
        if sample_image_path is None:
            pytest.skip("実サンプル画像が見つかりません。")

        cfg = PipelineConfig(
            output_dir=tmp_path / "results_real",
            num_iter=20,
            output_size=512,
            n_r=192,
            n_theta=540,
            show_diagnostic=True,
            save_intermediates=True,
        )
        pipeline = AstroLauePipeline(cfg)
        res = pipeline.run(image_source=sample_image_path)

        # 中心検出確認
        assert res.r0 > 10.0
        assert res.r_outer > res.r0 * 3.0

        # スポット検出確認
        assert len(res.spots) > 5

        # 診断プロット確認
        assert res.diagnostic_fig_path is not None
        assert res.diagnostic_fig_path.is_file()

        # CSV確認
        assert res.spots_csv_path is not None
        assert res.spots_csv_path.is_file()

    def test_wiener_deblur_mode(self, sample_image_path: Path | None, tmp_path: Path):
        """Wienerフィルタモードの実行テスト."""
        if sample_image_path is None:
            pytest.skip("実サンプル画像が見つかりません。")

        cfg = PipelineConfig(
            output_dir=tmp_path / "results_wiener",
            method="wiener",
            nsr=0.02,
            output_size=512,
            n_r=192,
            n_theta=540,
            show_diagnostic=True,
        )
        pipeline = AstroLauePipeline(cfg)
        res = pipeline.run(image_source=sample_image_path)

        assert res.restored_cartesian.shape == (512, 512)
        assert len(res.spots) > 0

    def test_cli_argument_parsing(self):
        """CLIパーサーの動作テスト."""
        args = parse_args([
            "--input", "share/画像260925.png",
            "--output", "./custom_results",
            "--iter", "30",
            "--blur-len", "11.5",
            "--method", "wiener",
            "--no-diagnostic",
        ])
        assert args.input == "share/画像260925.png"
        assert args.output == "./custom_results"
        assert args.iter == 30
        assert args.blur_len == 11.5
        assert args.method == "wiener"
        assert args.show_diagnostic is False
