"""AstroLaue パイプラインオーケストレーター.

GUIキャプチャから幾何規格化、極座標展開、Richardson-Lucy逆畳み込み、
直交復元、スポット解析、物理妥当性検証、透過PNG出力、バッチ一括処理までを統括します。
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
import shutil
from typing import Optional, Dict, Any, List, Literal

import cv2
import numpy as np

from astrolaue.capture import load_image, capture_window, extract_diffraction_roi
from astrolaue.normalize import detect_center_and_radius, normalize_geometry, subtract_background
from astrolaue.polar import cartesian_to_polar, polar_to_cartesian
from astrolaue.deblur import (
    estimate_blur_length,
    create_psf_1d,
    estimate_asymmetric_psf_ensemble,
    richardson_lucy_circular,
    wiener_deblur_circular,
)
from astrolaue.analyze import analyze_diffraction_spots, SpotResult
from astrolaue.visualize import plot_diagnostic_figure, plot_batch_summary_figure, plot_comparison_figure
from astrolaue.validate import validate_physical_consistency, compute_residual_map, PhysicsValidationReport
from astrolaue.export import (
    export_publication_rgba,
    export_contrast_optimized_image,
    export_linear_image,
    export_three_way_comparison,
    safe_imwrite,
)
from astrolaue.clipboard import copy_image_to_clipboard

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """パイプライン実行設定."""

    output_dir: Path = Path("./results")
    num_iter: int = 30
    blur_length: Optional[float] = None
    kernel_type: Literal["rect", "gaussian", "trapezoid", "emg", "skew_normal", "asymmetric_shoulder", "auto"] = "rect"
    emg_tau: Optional[float] = None
    skew_alpha: Optional[float] = None
    method: Literal["rl", "wiener"] = "rl"
    damping: float = 0.005
    nsr: float = 0.01
    output_size: int = 1024
    n_r: int = 384
    n_theta: int = 1080
    bg_method: str = "tophat"
    bg_kernel_size: int = 51
    r_inner_ratio: float = 1.035
    r_outer_ratio: float = 0.995
    taper_width: int = 6
    spot_min_sigma: float = 2.0
    spot_max_sigma: float = 12.0
    spot_threshold: Optional[float] = None
    spot_max_spots: int = 150
    show_diagnostic: bool = True
    save_intermediates: bool = True
    validate_physics: bool = True
    export_transparent: bool = True
    export_linear: bool = True
    export_contrast_optimized: bool = True
    export_asinh: bool = True
    asinh_beta: Optional[float] = None
    export_comparison: bool = True


@dataclass
class PipelineResult:
    """パイプライン実行結果."""

    cropped_image: np.ndarray
    normalized_image: np.ndarray
    bg_removed_image: np.ndarray
    polar_before: np.ndarray
    polar_after: np.ndarray
    restored_cartesian: np.ndarray
    center: tuple[float, float]
    r0: float
    r_outer: float
    blur_length: float
    spots: List[SpotResult]
    representative_profile: Dict[str, Any]
    diagnostic_fig_path: Optional[Path] = None
    spots_csv_path: Optional[Path] = None
    transparent_png_path: Optional[Path] = None
    linear_png_path: Optional[Path] = None
    contrast_optimized_png_path: Optional[Path] = None
    contrast_pseudo_png_path: Optional[Path] = None
    contrast_comparison_path: Optional[Path] = None
    history_png_path: Optional[Path] = None
    comparison_fig_path: Optional[Path] = None
    physics_report: Optional[PhysicsValidationReport] = None
    residual_polar: Optional[np.ndarray] = None
    psf_params: Optional[Dict[str, Any]] = None


class AstroLauePipeline:
    """小型ラウエ画像復元パイプラインクラス."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        image_source: Optional[str | Path | np.ndarray] = None,
        live: bool = False,
        window_title: Optional[str] = None,
        target_mode: str = "auto",
    ) -> PipelineResult:
        """単一画像に対するパイプラインを実行します."""
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        # ステップ 1: 入力画像取得
        if live:
            logger.info("ライブキャプチャを実行中 (タイトル指定: %s, モード: %s)...", window_title, target_mode)
            raw_image = capture_window(window_title_pattern=window_title, target_mode=target_mode)
        elif image_source is not None:
            if isinstance(image_source, (str, Path)):
                logger.info("画像ファイルを読み込み中: %s", image_source)
                raw_image = load_image(image_source)
            elif isinstance(image_source, np.ndarray):
                raw_image = image_source.copy()
            else:
                raise TypeError(f"不正な image_source タイプです: {type(image_source)}")
        else:
            raise ValueError("image_source または live=True のいずれかを指定してください。")

        # ステップ 2: 回折像ROI切り出し
        cropped_roi, bbox = extract_diffraction_roi(raw_image)
        logger.info("ROI切り出し完了: %s, shape=%s", bbox, cropped_roi.shape)

        # グレースケール化
        if len(cropped_roi.shape) == 3:
            gray_crop = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_crop = cropped_roi.copy()

        # ステップ 3: 中心検出 & 半径計測
        center_crop, r0, r_outer = detect_center_and_radius(gray_crop)
        logger.info("中心検出: center=%s, r0=%.2f, r_outer=%.2f", center_crop, r0, r_outer)

        # ステップ 4: 幾何規格化 (キャンバス中心への移動)
        norm_img, norm_center = normalize_geometry(
            gray_crop,
            center=center_crop,
            output_size=cfg.output_size,
        )
        norm_r0 = r0
        norm_r_outer = r_outer

        # ステップ 5: 背景光除去
        bg_subtracted = subtract_background(
            norm_img,
            method=cfg.bg_method,
            kernel_size=cfg.bg_kernel_size,
        )

        # ステップ 6: 極座標展開 (r in [r_inner_ratio * r0, r_outer_ratio * r_outer])
        # 低角ピーク救済 (r_inner を 1.035 * r0 へ拡大) および外周ピーク救済 (0.995 * r_outer)
        r_inner = norm_r0 * cfg.r_inner_ratio
        r_outer_crop = min(norm_r_outer * cfg.r_outer_ratio, (cfg.output_size / 2.0) - 2.0)
        r_range = (float(r_inner), float(r_outer_crop))

        polar_before = cartesian_to_polar(
            bg_subtracted,
            center=norm_center,
            r_range=r_range,
            n_r=cfg.n_r,
            n_theta=cfg.n_theta,
        )

        # 内周境界のテーパー保護処理 (コサイン半テーパー / Tukey窓)
        # ビームストッパー直近の低角ピークを保護しつつ、内周端の急峻なカットによる逆畳み込みリンギングを抑制
        if cfg.taper_width > 0 and cfg.taper_width < cfg.n_r:
            w_tap = cfg.taper_width
            taper_curve = 0.5 * (1.0 - np.cos(np.pi * np.arange(w_tap, dtype=np.float32) / w_tap))
            polar_before[:w_tap, :] *= taper_curve[:, None]

        # ステップ 7: ブラー長推定 & PSF生成 (非対称アンサンブル高速同定対応)
        psf_meta: Dict[str, Any] = {}
        if cfg.kernel_type in ["auto", "emg", "skew_normal"]:
            if cfg.blur_length is not None:
                fallback_b = float(cfg.blur_length)
            else:
                fallback_b = estimate_blur_length(polar_before)

            psf, asym_params = estimate_asymmetric_psf_ensemble(
                polar_image=polar_before,
                model_type=cfg.kernel_type if cfg.kernel_type in ["emg", "skew_normal"] else "auto",
                fallback_blur=fallback_b,
            )
            blur_len = asym_params.blur_length
            psf_meta = {
                "kernel_type": asym_params.kernel_type,
                "sigma": asym_params.sigma,
                "tau": asym_params.tau,
                "alpha": asym_params.alpha,
                "blur_length": asym_params.blur_length,
                "r2_score": asym_params.r2_score,
                "num_spots_used": asym_params.num_spots_used,
            }
            logger.info(
                "非対称アンサンブルPSFを適用: モデル=%s (R²=%.3f, σ=%.2f, τ=%s, α=%s)",
                asym_params.kernel_type, asym_params.r2_score, asym_params.sigma,
                f"{asym_params.tau:.2f}" if asym_params.tau is not None else "None",
                f"{asym_params.alpha:.2f}" if asym_params.alpha is not None else "None",
            )
        else:
            if cfg.blur_length is None:
                blur_len = estimate_blur_length(polar_before)
            else:
                blur_len = float(cfg.blur_length)
            psf = create_psf_1d(blur_length=blur_len, kernel_type=cfg.kernel_type)
            psf_meta = {"kernel_type": cfg.kernel_type, "blur_length": blur_len}
            logger.info("使用ブラー長: L_theta = %.2f ピクセル (kernel=%s)", blur_len, cfg.kernel_type)

        # ステップ 8: 方位角逆畳み込み復元 (極座標)
        if cfg.method == "rl":
            polar_after = richardson_lucy_circular(
                polar_before,
                psf_1d=psf,
                num_iter=cfg.num_iter,
                damping=cfg.damping,
            )
            method_label = "Richardson-Lucy"
        elif cfg.method == "wiener":
            polar_after = wiener_deblur_circular(
                polar_before,
                psf_1d=psf,
                nsr=cfg.nsr,
            )
            method_label = "Wiener"
        else:
            raise ValueError(f"未知の復元手法です: {cfg.method}")

        # ステップ 9: 直交座標系への逆極座標変換
        restored_cart = polar_to_cartesian(
            polar_after,
            output_shape=(cfg.output_size, cfg.output_size),
            center=norm_center,
            r_range=r_range,
        )

        # ステップ 10: スポット検出 & FWHM・強度解析
        spots, rep_profile = analyze_diffraction_spots(
            cart_before=bg_subtracted,
            cart_after=restored_cart,
            polar_before=polar_before,
            polar_after=polar_after,
            center=norm_center,
            r_range=r_range,
            min_sigma=cfg.spot_min_sigma,
            max_sigma=cfg.spot_max_sigma,
            threshold=cfg.spot_threshold,
            max_spots=cfg.spot_max_spots,
        )

        # ステップ 11: 課題 1 物理妥当性の検証
        physics_report = None
        residual_polar = None
        if cfg.validate_physics:
            physics_report = validate_physical_consistency(
                polar_raw=polar_before,
                polar_restored=polar_after,
                psf_1d=psf,
                spots=spots,
            )
            residual_polar, _, _ = compute_residual_map(polar_before, polar_after, psf)

        # ステップ 12: 課題 2 提出用透過PNG (RGBA) 出力 & 履歴蓄積 & クリップボードコピー
        transparent_png_path = None
        history_png_path = None
        if cfg.export_transparent:
            transparent_png_path = cfg.output_dir / "restored_transparent.png"
            export_publication_rgba(
                image=restored_cart,
                center=norm_center,
                r_inner=norm_r0,
                r_outer=norm_r_outer,
                output_path=transparent_png_path,
                export_16bit=True,
                tight_crop=True,
                use_asinh=cfg.export_asinh,
                asinh_beta=cfg.asinh_beta,
            )

            # 履歴フォルダへのタイムスタンプ付き保存 (1代前で消えず永続的に蓄積)
            # ユーザー要望: 残す写真は restored_transparent.png だけで問題なし
            history_dir = cfg.output_dir / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            ts_str = time.strftime("%Y%m%d_%H%M%S")
            target_hist = history_dir / f"restored_{ts_str}.png"
            try:
                target_hist.write_bytes(transparent_png_path.read_bytes())
                history_png_path = target_hist
                logger.info("履歴画像を保存しました: %s", history_png_path)
            except Exception as e_copy:
                logger.warning("履歴画像の保存中に警告が発生しました: %s", e_copy)

            # クリップボード自動コピー (Windows / Linux)
            copied = copy_image_to_clipboard(transparent_png_path)
            if copied:
                logger.info("復元画像 (%s) をクリップボードにコピーしました。", transparent_png_path.name)

        # ステップ 12b: 元画像形式準拠の線形復元画像 & コントラスト最適化画像 (HDR) & 3並列比較
        linear_png_path = None
        if cfg.export_linear:
            linear_png_path = cfg.output_dir / "restored_linear.png"
            export_linear_image(
                image=restored_cart,
                center=norm_center,
                r_inner=norm_r0,
                r_outer=norm_r_outer,
                output_path=linear_png_path,
            )

        contrast_optimized_png_path = None
        contrast_pseudo_png_path = None
        contrast_comparison_path = None
        if cfg.export_contrast_optimized:
            contrast_opt_path = cfg.output_dir / "restored_contrast_optimized.png"
            contrast_optimized_png_path, contrast_pseudo_png_path = export_contrast_optimized_image(
                image=restored_cart,
                center=norm_center,
                r_inner=norm_r0,
                r_outer=norm_r_outer,
                output_path=contrast_opt_path,
                asinh_beta=cfg.asinh_beta if cfg.asinh_beta is not None else 6.0,
                export_pseudo_color=True,
            )

            # 3並列比較図の生成 [元画像 | 線形復元 | コントラスト最適化]
            contrast_comparison_path = cfg.output_dir / "contrast_comparison.png"
            img_lin = (
                cv2.imread(str(linear_png_path), cv2.IMREAD_GRAYSCALE)
                if linear_png_path and linear_png_path.exists()
                else norm_img
            )
            img_opt = (
                cv2.imread(str(contrast_optimized_png_path), cv2.IMREAD_GRAYSCALE)
                if contrast_optimized_png_path and contrast_optimized_png_path.exists()
                else norm_img
            )
            export_three_way_comparison(
                original_image=norm_img,
                linear_image=img_lin,
                optimized_image=img_opt,
                output_path=contrast_comparison_path,
            )

        # 中間成果物 & 診断プロットの保存
        diag_path = None
        csv_path = None

        if cfg.show_diagnostic:
            diag_path = cfg.output_dir / "diagnostic_report.png"
            plot_diagnostic_figure(
                cropped_orig=cropped_roi,
                center_orig=center_crop,
                r0=r0,
                r_outer=r_outer,
                polar_before=polar_before,
                polar_after=polar_after,
                restored_cart=restored_cart,
                spots=spots,
                profile_data=rep_profile,
                save_path=diag_path,
                blur_length=blur_len,
                num_iter=cfg.num_iter if cfg.method == "rl" else 1,
                method_name=method_label,
                physics_report=physics_report,
                residual_polar=residual_polar,
            )

        if cfg.save_intermediates:
            csv_path = cfg.output_dir / "detected_spots.csv"
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "spot_id", "x", "y", "r_px", "theta_deg",
                        "peak_intensity", "integrated_intensity",
                        "fwhm_before_px", "fwhm_after_px", "fwhm_reduction_percent", "sigma"
                    ])
                    for s in spots:
                        writer.writerow([
                            s.spot_id, f"{s.x:.2f}", f"{s.y:.2f}", f"{s.r:.2f}", f"{s.theta_deg:.2f}",
                            f"{s.peak_intensity:.4f}", f"{s.integrated_intensity:.4f}",
                            f"{s.fwhm_before:.2f}", f"{s.fwhm_after:.2f}", f"{s.fwhm_reduction_ratio:.1f}",
                            f"{s.sigma:.2f}"
                        ])
            except PermissionError as pe:
                logger.warning("detected_spots.csv が別アプリ(Excel等)で開かれているため上書きできませんでした: %s", pe)
                alt_csv = cfg.output_dir / f"detected_spots_{time.strftime('%H%M%S')}.csv"
                try:
                    with open(alt_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "spot_id", "x", "y", "r_px", "theta_deg",
                            "peak_intensity", "integrated_intensity",
                            "fwhm_before_px", "fwhm_after_px", "fwhm_reduction_percent", "sigma"
                        ])
                        for s in spots:
                            writer.writerow([
                                s.spot_id, f"{s.x:.2f}", f"{s.y:.2f}", f"{s.r:.2f}", f"{s.theta_deg:.2f}",
                                f"{s.peak_intensity:.4f}", f"{s.integrated_intensity:.4f}",
                                f"{s.fwhm_before:.2f}", f"{s.fwhm_after:.2f}", f"{s.fwhm_reduction_ratio:.1f}",
                                f"{s.sigma:.2f}"
                            ])
                    csv_path = alt_csv
                except Exception:
                    pass

            def to_u8(arr: np.ndarray, use_asinh_stretch: bool = False) -> np.ndarray:
                arr_f = np.maximum(arr.astype(np.float32), 0.0)
                mx = float(arr_f.max())
                if mx <= 0:
                    return np.zeros_like(arr, dtype=np.uint8)
                if use_asinh_stretch:
                    pos = arr_f[arr_f > 0]
                    if len(pos) > 100:
                        med = float(np.median(pos))
                        mad = float(np.median(np.abs(pos - med)))
                        beta = max(3.0 * 1.4826 * mad, 0.03 * mx)
                    else:
                        beta = 0.03 * mx
                    denom = float(np.arcsinh(mx / max(beta, 1e-6)))
                    scaled = (np.arcsinh(arr_f / max(beta, 1e-6)) / denom) * 255.0 if denom > 0 else (arr_f / mx * 255.0)
                else:
                    scaled = (arr_f / mx * 255.0)
                return np.clip(scaled, 0, 255).astype(np.uint8)

            safe_imwrite(cfg.output_dir / "cropped_roi.png", cropped_roi)
            safe_imwrite(cfg.output_dir / "normalized_canvas.png", norm_img)
            safe_imwrite(cfg.output_dir / "bg_removed.png", to_u8(bg_subtracted))
            safe_imwrite(cfg.output_dir / "polar_before.png", to_u8(polar_before))
            # タイトクロップされた直交復元画像の保存 (余白ゼロ化 & asinh微弱ピーク強調)
            crop_r = int(round(norm_r_outer * 1.06))
            xc_i, yc_i = int(round(norm_center[0])), int(round(norm_center[1]))
            x1 = max(0, xc_i - crop_r)
            y1 = max(0, yc_i - crop_r)
            x2 = min(cfg.output_size, xc_i + crop_r)
            y2 = min(cfg.output_size, yc_i + crop_r)
            side = min(x2 - x1, y2 - y1)
            restored_tight = restored_cart[y1 : y1 + side, x1 : x1 + side]
            safe_imwrite(cfg.output_dir / "restored_cartesian.png", to_u8(restored_tight, use_asinh_stretch=cfg.export_asinh))

            if residual_polar is not None:
                safe_imwrite(cfg.output_dir / "residual_polar.png", to_u8(residual_polar))

        # ステップ 13: ビフォーアフター検証比較画像の保存
        comparison_png_path = None
        if cfg.export_comparison:
            comparison_png_path = cfg.output_dir / "comparison_before_after.png"
            plot_comparison_figure(
                cropped_orig=cropped_roi,
                restored_cart=restored_cart,
                center_norm=norm_center,
                r_outer=norm_r_outer,
                spots=spots,
                save_path=comparison_png_path,
                blur_length=blur_len,
                physics_report=physics_report,
            )

        logger.info("AstroLaue パイプライン処理が正常に完了しました。")

        return PipelineResult(
            cropped_image=cropped_roi,
            normalized_image=norm_img,
            bg_removed_image=bg_subtracted,
            polar_before=polar_before,
            polar_after=polar_after,
            restored_cartesian=restored_cart,
            center=norm_center,
            r0=norm_r0,
            r_outer=norm_r_outer,
            blur_length=blur_len,
            spots=spots,
            representative_profile=rep_profile,
            diagnostic_fig_path=diag_path,
            spots_csv_path=csv_path,
            transparent_png_path=transparent_png_path,
            linear_png_path=linear_png_path,
            contrast_optimized_png_path=contrast_optimized_png_path,
            contrast_pseudo_png_path=contrast_pseudo_png_path,
            contrast_comparison_path=contrast_comparison_path,
            history_png_path=history_png_path,
            comparison_fig_path=comparison_png_path,
            physics_report=physics_report,
            residual_polar=residual_polar,
            psf_params=psf_meta,
        )

    def run_batch(
        self,
        batch_dir: str | Path,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp"),
    ) -> Path:
        """指定ディレクトリ内の全画像を探索して一括処理し、summary_report.csvおよび比較図を出力します.

        Args:
            batch_dir: 画像が格納されたディレクトリパス
            extensions: 探索対象の拡張子

        Returns:
            summary_csv_path: 生成された summary_report.csv のパス
        """
        p_dir = Path(batch_dir)
        if not p_dir.is_dir():
            # フォールバック (例: /share が無ければ ./share)
            if Path("share").is_dir():
                logger.warning("指定パス '%s' が見つからないため、'share' を使用します。", p_dir)
                p_dir = Path("share")
            else:
                raise FileNotFoundError(f"ディレクトリが見つかりません: {batch_dir}")

        image_files = sorted([
            f for f in p_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ])

        if not image_files:
            raise FileNotFoundError(f"ディレクトリ内に画像ファイルが見つかりません: {p_dir}")

        logger.info("バッチ処理開始: %d 個の画像を検出しました (%s)", len(image_files), p_dir)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        batch_plot_data = []

        for img_path in image_files:
            logger.info("--- バッチ処理中: %s ---", img_path.name)
            sub_out = self.config.output_dir / img_path.stem
            sub_out.mkdir(parents=True, exist_ok=True)

            # 各画像用にサブディレクトリ設定 (全設定を確実に継承)
            sub_cfg = replace(self.config, output_dir=sub_out)
            sub_pipe = AstroLauePipeline(sub_cfg)

            t0 = time.time()
            try:
                res = sub_pipe.run(image_source=img_path)
                elapsed = time.time() - t0

                reductions = [s.fwhm_reduction_ratio for s in res.spots if s.fwhm_before > 0]
                avg_red = float(np.mean(reductions)) if reductions else 0.0

                flux_ratio = res.physics_report.mean_flux_ratio if res.physics_report else 1.0
                c_shift = res.physics_report.mean_centroid_shift_px if res.physics_report else 0.0
                rmse = res.physics_report.residual_rmse if res.physics_report else 0.0

                summary_rows.append({
                    "image_name": img_path.name,
                    "status": "SUCCESS",
                    "center_x": f"{res.center[0]:.2f}",
                    "center_y": f"{res.center[1]:.2f}",
                    "r0_px": f"{res.r0:.2f}",
                    "r_outer_px": f"{res.r_outer:.2f}",
                    "blur_length_px": f"{res.blur_length:.2f}",
                    "num_spots": len(res.spots),
                    "avg_fwhm_reduction_pct": f"{avg_red:.1f}",
                    "mean_flux_ratio": f"{flux_ratio:.4f}",
                    "mean_centroid_shift_px": f"{c_shift:.3f}",
                    "residual_rmse": f"{rmse:.4f}",
                    "time_sec": f"{elapsed:.2f}",
                    "error_msg": "",
                })

                batch_plot_data.append({
                    "image_name": img_path.name,
                    "status": "SUCCESS",
                    "cropped_image": res.cropped_image,
                    "restored_image": res.restored_cartesian,
                    "profile_data": res.representative_profile,
                    "blur_length": res.blur_length,
                    "avg_fwhm_reduction": avg_red,
                    "num_spots": len(res.spots),
                })
            except Exception as ex:
                elapsed = time.time() - t0
                logger.exception("画像 '%s' の処理に失敗しました: %s", img_path.name, ex)
                summary_rows.append({
                    "image_name": img_path.name,
                    "status": "FAILED",
                    "center_x": "",
                    "center_y": "",
                    "r0_px": "",
                    "r_outer_px": "",
                    "blur_length_px": "",
                    "num_spots": 0,
                    "avg_fwhm_reduction_pct": "0.0",
                    "mean_flux_ratio": "",
                    "mean_centroid_shift_px": "",
                    "residual_rmse": "",
                    "time_sec": f"{elapsed:.2f}",
                    "error_msg": str(ex),
                })
                batch_plot_data.append({
                    "image_name": img_path.name,
                    "status": "FAILED",
                })

        # summary_report.csv の保存
        csv_path = self.config.output_dir / "summary_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "image_name", "status", "center_x", "center_y", "r0_px", "r_outer_px",
                "blur_length_px", "num_spots", "avg_fwhm_reduction_pct",
                "mean_flux_ratio", "mean_centroid_shift_px", "residual_rmse",
                "time_sec", "error_msg",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        # batch_summary_plot.png の保存
        plot_path = self.config.output_dir / "batch_summary_plot.png"
        plot_batch_summary_figure(batch_plot_data, plot_path)

        logger.info("バッチ処理完了: サマリーCSV=%s, 一覧比較図=%s", csv_path, plot_path)
        return csv_path
