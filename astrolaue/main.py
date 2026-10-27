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
        "--live", "--watch", "-w",
        action="store_true",
        dest="live",
        help="リアルタイム・ライブ監視モード (生データフォルダ新着CSV常時自動復元 + ホットキー/Enter手動キャプチャ)",
    )
    parser.add_argument(
        "--snapshot", "--capture-once",
        action="store_true",
        help="単発画面キャプチャモード (手前のウィンドウまたは画面全体から1回取得して終了)",
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default=None,
        help="生データ (.CSV) 監視フォルダパス (省略時は Data/2610 等を自動探索)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Rigaku sLaue 極座標生データ (.CSV) ファイルパス",
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
        "--align",
        action="store_true",
        help="結晶方位アライメントモード (主要晶帯軸 (001) の自動同定と2軸ゴニオ補正角を推定)",
    )
    parser.add_argument(
        "--camera-length", "-L",
        type=float,
        default=27.0,
        help="試料―検出器間距離 L [mm] (アライメント用、sLaue 実機幾何: 27.0)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=0.05,
        help="検出器ピクセルサイズ [mm/pix] (画像入力時アライメント用、デフォルト: 0.05)",
    )
    parser.add_argument(
        "--stage-order",
        choices=["XY", "YX"],
        default="XY",
        help="ゴニオメーターステージ軸順序 ('XY': 下X上Y, 'YX': 下Y上X) (デフォルト: XY)",
    )
    parser.add_argument(
        "--sign-x",
        type=int,
        choices=[1, -1],
        default=1,
        help="X軸回転の符号係数 (デフォルト: 1)",
    )
    parser.add_argument(
        "--sign-y",
        type=int,
        choices=[1, -1],
        default=-1,
        help="Y軸回転の符号係数 (sLaue 実機ステージ整合: -1)",
    )
    parser.add_argument(
        "--saturation-limit",
        type=float,
        default=7.72e6,
        help="検出器の飽和カウント上限値 (デフォルト: 7,720,000 counts)",
    )
    parser.add_argument(
        "--target-exposure-ratio",
        type=float,
        default=0.80,
        help="露光最適化の目標飽和比率 (デフォルト: 0.80 = 上限80%)",
    )
    parser.add_argument(
        "--no-optimize-exposure",
        action="store_false",
        dest="optimize_exposure",
        help="露光時間最適化解析を無効化する",
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
    if getattr(parsed, "live", False) and parsed.output == "./results":
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
        optimize_exposure=getattr(parsed, "optimize_exposure", True),
        saturation_limit=getattr(parsed, "saturation_limit", 7.72e6),
        target_exposure_ratio=getattr(parsed, "target_exposure_ratio", 0.80),
    )
    pipeline = AstroLauePipeline(config)

    # 0. 結晶方位アライメントモード
    if parsed.align:
        from astrolaue.alignment import (
            LaueAlignmentConfig,
            solve_laue_alignment,
            plot_alignment_diagnostics,
            generate_synthetic_laue_sample,
        )
        align_cfg = LaueAlignmentConfig(
            camera_length_mm=parsed.camera_length,
            pixel_size_mm=parsed.pixel_size,
            stage_order=parsed.stage_order,
            sign_x=parsed.sign_x,
            sign_y=parsed.sign_y,
        )
        out_dir = Path(parsed.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        diag_path = out_dir / "laue_alignment_diagnostic.png"

        print("=" * 75)
        print("  AstroLaue: 結晶方位アライメント (晶帯軸 (001) 自動同定 & ゴニオ補正角推定)")
        print("=" * 75)
        print(f"・カメラ長 L: {align_cfg.camera_length_mm:.1f} mm")
        print(f"・ピクセルサイズ: {align_cfg.pixel_size_mm:.4f} mm/pix")
        print(f"・ステージ構成: {align_cfg.stage_order} (sign_x={align_cfg.sign_x}, sign_y={align_cfg.sign_y})")

        if parsed.input:
            input_path = Path(parsed.input)
            print(f"・解析対象画像: {input_path.resolve()}")
            res = solve_laue_alignment(input_path, config=align_cfg)
        else:
            print("・入力画像未指定のため、合成正方晶シミュレーションを実行します。")
            spots_xy, intens, meta = generate_synthetic_laue_sample(
                true_tilt_x_deg=6.0,
                true_tilt_y_deg=-4.5,
                camera_length_mm=align_cfg.camera_length_mm,
                lattice_type="tetragonal",
            )
            res = solve_laue_alignment(spots_xy, intensities=intens, config=align_cfg)

        best = res.best_candidate
        print("\n" + "=" * 75)
        print(" 【主軸同定 & ゴニオ補正角 推定結果】")
        print("=" * 75)
        print(f"検出スポット数: {len(res.spots_xy_mm)} 点")
        print(f"同定主要晶帯数: {len(res.zones)} 本")
        for z in res.zones:
            print(f"  - Zone #{z.line_id}: a={z.a:+.3f}, b={z.b:+.3f}, c={z.c:+.3f} | Score={z.score:5.1f} | Inliers={z.num_inliers:2d}")

        print(f"\n同定された主軸 (001) 候補 (上位 {len(res.candidates)} 件):")
        for c in res.candidates:
            print(f"  [第{c.rank}候補] グノモン(u, v)=({c.u:+.4f}, {c.v:+.4f}) | 検出器(X, Y)=({c.detector_x_mm:+.2f}, {c.detector_y_mm:+.2f}) mm")
            print(f"             補正角: Δθx = {c.delta_theta_x_deg:+.3f}°,  Δθy = {c.delta_theta_y_deg:+.3f}° (全傾き={c.total_tilt_deg:.2f}°) | 信頼度={c.score:6.1f}")

        print("-" * 75)
        print(f"★ 推奨ゴニオメーター補正回転角:  Δθx = {best.delta_theta_x_deg:+.3f}°,  Δθy = {best.delta_theta_y_deg:+.3f}°")
        print(f"★ 現在の結晶主軸傾き角 ψ:       {best.total_tilt_deg:.3f}°")
        print(f"★ 解析処理時間:                 {res.elapsed_ms:.1f} ms")
        print("=" * 75)

        plot_alignment_diagnostics(res, save_path=diag_path, show=False)
        print(f"アライメント診断図を保存しました: {diag_path.resolve()}\n")
        return 0

    # 1. リアルタイム・ライブ監視モード (生データ新着CSV監視 + ホットキー/Enter監視)
    # --live, --watch が指定された場合、または入力・バッチ指定がない場合のデフォルト動作
    input_path = parsed.csv or parsed.input
    is_live_mode = parsed.live or getattr(parsed, "watch", False)
    if not is_live_mode and input_path is None and parsed.batch_dir is None and not parsed.align and not getattr(parsed, "snapshot", False):
        logger.info("入力指定がないため、デフォルトのリアルタイム・ライブ監視モードを起動します。")
        is_live_mode = True

    if is_live_mode:
        from astrolaue.watcher import ResidentWatcher
        logger.info("リアルタイム・ライブ監視モードを起動します (出力先: %s, 監視先: %s)", parsed.output, parsed.watch_dir or "自動探索")
        watcher = ResidentWatcher(
            config=config,
            output_dir=parsed.output,
            watch_dir=parsed.watch_dir,
            window_title=parsed.window_title,
            monitor_index=parsed.monitor,
            target_mode=parsed.target_mode,
            camera_length_mm=getattr(parsed, "camera_length", 27.0),
            pixel_size_mm=getattr(parsed, "pixel_size", 0.05),
            stage_order=getattr(parsed, "stage_order", "XY"),
            sign_x=getattr(parsed, "sign_x", 1),
            sign_y=getattr(parsed, "sign_y", -1),
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

    # 3. 単一画像 / CSV生データ / ライブキャプチャモード
    input_path = parsed.csv or parsed.input
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
            live=getattr(parsed, "snapshot", False),
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
        if res.alignment_result:
            best = res.alignment_result.best_candidate
            print(f"・【推奨ゴニオ補正角】: Δθx = {best.delta_theta_x_deg:+.2f}°, Δθy = {best.delta_theta_y_deg:+.2f}° (全傾き角: {best.total_tilt_deg:.2f}°)")
            if res.alignment_fig_path:
                print(f"・アライメント診断図: {res.alignment_fig_path}")
        print("・クリップボード: 提出用透過PNGをクリップボードに自動格納しました")
        print("==========================================\n")

        if res.exposure_result:
            print(res.exposure_result.format_section() + "\n")

        return 0
    except Exception as e:
        logger.exception("パイプライン実行中にエラーが発生しました: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
