"""AstroLaue コマンドラインインターフェース (CLI)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from astrolaue.pipeline import AstroLauePipeline, PipelineConfig


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="AstroLaue",
        description="AstroLaue: 小型ラウエ画像復元パイプライン (円周方向モーションブラー逆畳み込み)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="単一入力画像パス (例: share/画像260925.png または /share/...)",
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default=None,
        help="一括処理対象の画像ディレクトリパス (例: /share または share/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="出力ディレクトリ (デフォルト: ./results/)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=30,
        help="Richardson-Lucy逆畳み込みの反復回数 (デフォルト: 30)",
    )
    parser.add_argument(
        "--blur-len", "-b",
        type=float,
        default=None,
        help="方位角ブラー長 (ピクセル)。省略時は自己相関から自動推定",
    )
    parser.add_argument(
        "--kernel-type",
        choices=["rect", "gaussian", "trapezoid"],
        default="rect",
        help="PSFカーネル形状 (デフォルト: rect)",
    )
    parser.add_argument(
        "--method",
        choices=["rl", "wiener"],
        default="rl",
        help="復元アルゴリズム ('rl': Richardson-Lucy, 'wiener': Wiener) (デフォルト: rl)",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.005,
        help="Richardson-Lucyのダンピング係数 (デフォルト: 0.005)",
    )
    parser.add_argument(
        "--nsr",
        type=float,
        default=0.01,
        help="WienerフィルタのNSR (Noise-to-Signal Ratio) (デフォルト: 0.01)",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=1024,
        help="幾何規格化キャンバスのピクセルサイズ (デフォルト: 1024)",
    )
    parser.add_argument(
        "--show-diagnostic",
        action="store_true",
        default=True,
        help="5要素診断レポート図 (PNG) を出力する (デフォルト: True)",
    )
    parser.add_argument(
        "--no-diagnostic",
        dest="show_diagnostic",
        action="store_false",
        help="診断レポート図の出力を無効化する",
    )
    parser.add_argument(
        "--export-transparent",
        action="store_true",
        default=True,
        help="学術提出用透過PNG (RGBA, restored_transparent.png) を出力する (デフォルト: True)",
    )
    parser.add_argument(
        "--validate-physics",
        action="store_true",
        default=True,
        help="物理的妥当性検証 (光量保存性・重心不変性・残差マップ) を実行する (デフォルト: True)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="画面キャプチャモード (手前のウィンドウまたは画面全体から1回自動取得して復元)",
    )
    parser.add_argument(
        "--watch", "--resident",
        action="store_true",
        dest="watch",
        help="常駐ホットキー監視モード (F9 または Ctrl+Shift+L でいつでも即時キャプチャ＆復元)",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default=None,
        help="キャプチャ対象のウィンドウタイトル (省略時は特定アプリに依存せず手前のウィンドウまたは画面全体を取得)",
    )
    parser.add_argument(
        "--target-mode",
        choices=["auto", "screen", "active"],
        default="auto",
        help="キャプチャ対象 ('auto': 手前ウィンドウまたは画面, 'screen': 画面全体, 'active': アクティブウィンドウ)",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="キャプチャ対象モニタ番号 (デフォルト: 1=プライマリモニタ, 0=全マルチ画面)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログ出力レベル",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    parsed = parse_args(args)

    logging.basicConfig(
        level=getattr(logging, parsed.log_level.upper()),
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("astrolaue")

    # ライブまたは常駐モードの場合、未指定時の出力先を results_live に自動切り替え
    if (parsed.watch or parsed.live) and parsed.output == "./results":
        parsed.output = "./results_live"

    config = PipelineConfig(
        output_dir=Path(parsed.output),
        num_iter=parsed.iter,
        blur_length=parsed.blur_len,
        kernel_type=parsed.kernel_type,
        method=parsed.method,
        damping=parsed.damping,
        nsr=parsed.nsr,
        output_size=parsed.output_size,
        show_diagnostic=parsed.show_diagnostic,
        validate_physics=parsed.validate_physics,
        export_transparent=parsed.export_transparent,
    )

    pipeline = AstroLauePipeline(config)

    # 1. 常駐ホットキー監視モード
    if parsed.watch:
        from astrolaue.watcher import ResidentWatcher
        logger.info("常駐ホットキー監視モードを起動します (出力先: %s)", parsed.output)
        watcher = ResidentWatcher(
            config=config,
            output_dir=parsed.output,
            window_title=parsed.window_title,
            monitor_index=parsed.monitor,
            target_mode=parsed.target_mode,
        )
        watcher.run()
        return 0

    # 2. バッチモード
    if parsed.batch_dir is not None:
        target_dir = Path(parsed.batch_dir)
        if not target_dir.is_dir():
            if Path("share").is_dir():
                target_dir = Path("share")
            elif Path("/share").is_dir():
                target_dir = Path("/share")
        logger.info("バッチ実行を開始します: ディレクトリ=%s", target_dir)
        try:
            summary_csv = pipeline.run_batch(target_dir)
            print("\n==========================================")
            print("  AstroLaue バッチ処理完了サマリー  ")
            print("==========================================")
            print(f"・バッチ結果CSV: {summary_csv}")
            print(f"・バッチ一覧図 : {config.output_dir / 'batch_summary_plot.png'}")
            print("==========================================\n")
            return 0
        except Exception as e:
            logger.exception("バッチ処理中にエラーが発生しました: %s", e)
            return 1

    # 3. 単一画像 / ライブキャプチャモード
    input_path = parsed.input
    if not parsed.live and input_path is None:
        # デフォルト探索
        if Path("share/画像260925.png").is_file():
            input_path = "share/画像260925.png"
        elif Path("/share/画像260925.png").is_file():
            input_path = "/share/画像260925.png"
        elif Path("share").is_dir():
            for p in Path("share").glob("*.png"):
                input_path = str(p)
                break
        if input_path:
            logger.info("入力指定がないため、見つかった画像を使用します: %s", input_path)
        else:
            logger.error("入力画像が指定されておらず、デフォルト画像も見つかりません。--input または --batch-dir を指定してください。")
            return 1
    elif input_path is not None:
        # /share/... が渡されて実在せず、share/... がある場合のフォールバック
        p_in = Path(input_path)
        if not p_in.is_file() and str(input_path).startswith("/share"):
            rel_p = Path(str(input_path).lstrip("/"))
            if rel_p.is_file():
                input_path = str(rel_p)

    try:
        res = pipeline.run(
            image_source=input_path,
            live=parsed.live,
            window_title=parsed.window_title,
            target_mode=parsed.target_mode,
        )
        print("\n==========================================")
        print("  AstroLaue パイプライン実行結果サマリー  ")
        print("==========================================")
        print(f"・検出中心座標: ({res.center[0]:.2f}, {res.center[1]:.2f})")
        print(f"・基準白丸半径: {res.r0:.2f} px")
        print(f"・外周円盤半径: {res.r_outer:.2f} px")
        print(f"・適用ブラー長 (L_theta): {res.blur_length:.2f} px")
        print(f"・検出スポット総数: {len(res.spots)} 個")
        if res.spots:
            reductions = [s.fwhm_reduction_ratio for s in res.spots if s.fwhm_before > 0]
            avg_red = sum(reductions) / len(reductions) if reductions else 0.0
            print(f"・平均FWHM先鋭化率: {avg_red:.1f}%")

        if res.physics_report:
            print("\n【物理的・原理的妥当性検証】")
            print(f"・光量保存比 (Flux Ratio) : {res.physics_report.mean_flux_ratio:.4f} (判定: {'PASS' if res.physics_report.flux_conservation_passed else 'CHECK'})")
            print(f"・重心移動 (Centroid Shift): {res.physics_report.mean_centroid_shift_px:.3f} px (最大: {res.physics_report.max_centroid_shift_px:.3f} px, 判定: {'PASS' if res.physics_report.centroid_invariance_passed else 'CHECK'})")
            print(f"・再投影残差 (RMSE)        : {res.physics_report.residual_rmse:.4f} (相対: {res.physics_report.residual_mean_relative*100:.1f}%)")

        if res.comparison_fig_path:
            print(f"\n・検証用比較画像: {res.comparison_fig_path}")
        if res.transparent_png_path:
            print(f"・提出用透過PNG: {res.transparent_png_path}")
        if res.history_png_path:
            print(f"・履歴蓄積PNG  : {res.history_png_path}")
        if res.diagnostic_fig_path:
            print(f"・診断レポート図: {res.diagnostic_fig_path}")
        if res.spots_csv_path:
            print(f"・スポット解析CSV: {res.spots_csv_path}")
        print("・クリップボード: 提出用透過PNGをクリップボードに自動格納しました")
        print("==========================================\n")
        return 0
    except Exception as e:
        logger.exception("パイプライン実行中にエラーが発生しました: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
