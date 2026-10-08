"""モジュール 6: 診断可視化 & バッチ比較レポート.

以下の機能を提供します:
1. 5要素診断レポート図 (PNG) の描画・保存 (物理妥当性・残差情報付き)
2. バッチ処理の一覧比較図 (batch_summary_plot.png) の生成
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from astropy.visualization import AsinhStretch

import time
from astrolaue.analyze import SpotResult

logger = logging.getLogger(__name__)


def safe_savefig(fig: plt.Figure, save_path: str | Path, **kwargs) -> Path:
    """Windows でファイルがビューアで開かれていてもクラッシュしない安全な保存関数."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(str(p), **kwargs)
        return p
    except PermissionError as pe:
        logger.warning("画像が別アプリで開かれているため上書きできませんでした (%s): %s", p, pe)
        alt_p = p.with_name(f"{p.stem}_{time.strftime('%H%M%S')}{p.suffix}")
        try:
            fig.savefig(str(alt_p), **kwargs)
            logger.info("代替ファイル名として保存しました: %s", alt_p)
            return alt_p
        except Exception:
            return p


def plot_diagnostic_figure(
    cropped_orig: np.ndarray,
    center_orig: Tuple[float, float],
    r0: float,
    r_outer: float,
    polar_before: np.ndarray,
    polar_after: np.ndarray,
    restored_cart: np.ndarray,
    spots: List[SpotResult],
    profile_data: Dict[str, Any],
    save_path: str | Path | None = None,
    blur_length: float | None = None,
    num_iter: int = 25,
    method_name: str = "Richardson-Lucy",
    physics_report: Any | None = None,
    residual_polar: np.ndarray | None = None,
) -> plt.Figure:
    """5つの診断要素および物理検証指標を集約して描画・保存します.

    Args:
        cropped_orig: クロップされた元ROI画像
        center_orig: クロップ画像上の中心座標 (xc, yc)
        r0: 中心白丸半径
        r_outer: 外周円盤半径
        polar_before: 復元前の極座標展開画像 (n_r, n_theta)
        polar_after: 復元後の極座標展開画像 (n_r, n_theta)
        restored_cart: 復元後の直交画像
        spots: 検出スポットのリスト
        profile_data: 代表スポットのプロファイル辞書
        save_path: 保存先ファイルパス
        blur_length: 復元に用いたブラー長
        num_iter: 逆畳み込み反復数
        method_name: 手法名
        physics_report: 物理妥当性検証レポートオブジェクト
        residual_polar: 残差極座標画像

    Returns:
        fig: 作成された matplotlib Figure
    """
    fig = plt.figure(figsize=(16, 12), facecolor="#1e1e1e")
    
    title_suffix = ""
    if physics_report is not None:
        title_suffix = f" | Flux Ratio: {physics_report.mean_flux_ratio:.3f}, ΔCentroid: {physics_report.mean_centroid_shift_px:.2f}px"

    fig.suptitle(
        f"AstroLaue Diagnostic Report | {method_name} (iter={num_iter}, L={blur_length:.1f}px){title_suffix}",
        fontsize=15,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9], hspace=0.28, wspace=0.18)

    # 1. クロップ元画像 (検出中心表示)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#121212")
    if len(cropped_orig.shape) == 3:
        ax1.imshow(cropped_orig[..., ::-1])
    else:
        ax1.imshow(cropped_orig, cmap="gray")
    xc, yc = center_orig
    ax1.scatter([xc], [yc], color="red", marker="+", s=120, linewidths=2.5, label="Center (Beam Stop)")
    circle_inner = patches.Circle((xc, yc), r0, fill=False, edgecolor="cyan", linewidth=1.8, linestyle="--", label=f"Center r0={r0:.1f}px")
    circle_outer = patches.Circle((xc, yc), r_outer, fill=False, edgecolor="yellow", linewidth=1.5, linestyle=":", label=f"Outer r={r_outer:.1f}px")
    ax1.add_patch(circle_inner)
    ax1.add_patch(circle_outer)
    ax1.set_title("1. Cropped Original & Detected Center", color="#ffffff", fontsize=12, pad=6)
    ax1.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#e0e0e0", fontsize=9)
    ax1.tick_params(colors="#888888")

    # 4. 復元後の直交画像 (検出スポット表示)
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.set_facecolor("#121212")
    im4 = ax4.imshow(restored_cart, cmap="inferno")
    for s in spots[:60]:
        c_spot = patches.Circle((s.x, s.y), radius=max(s.sigma * 1.5, 4.0), fill=False, edgecolor="#00ffcc", linewidth=1.5)
        ax4.add_patch(c_spot)
        ax4.text(s.x + 5, s.y - 5, f"#{s.spot_id}", color="#00ffcc", fontsize=8, fontweight="bold")
    ax4.set_title(f"4. Restored Cartesian & Detected Spots (N={len(spots)})", color="#ffffff", fontsize=12, pad=6)
    ax4.tick_params(colors="#888888")
    cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(colors="#aaaaaa")

    # 2. 極座標展開画像 (復元前)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#121212")
    n_r, n_theta = polar_before.shape[:2]
    im2 = ax2.imshow(
        polar_before,
        cmap="inferno",
        aspect="auto",
        extent=[0, 360, n_r, 0],
    )
    ax2.set_title("2. Polar Projection (Before Deblur: Azimuthal Blur)", color="#ffffff", fontsize=12, pad=6)
    ax2.set_xlabel("Azimuth Angle θ [deg]", color="#e0e0e0", fontsize=10)
    ax2.set_ylabel("Radial Distance r [idx]", color="#e0e0e0", fontsize=10)
    ax2.tick_params(colors="#888888")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#aaaaaa")

    # 3. 復元後の極座標画像 (または残差マップ併記)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#121212")
    im3 = ax3.imshow(
        polar_after,
        cmap="inferno",
        aspect="auto",
        extent=[0, 360, n_r, 0],
    )
    ax3.set_title("3. Deblurred Polar Image (Sharpened Spots)", color="#ffffff", fontsize=12, pad=6)
    ax3.set_xlabel("Azimuth Angle θ [deg]", color="#e0e0e0", fontsize=10)
    ax3.set_ylabel("Radial Distance r [idx]", color="#e0e0e0", fontsize=10)
    ax3.tick_params(colors="#888888")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.ax.tick_params(colors="#aaaaaa")

    # 5. 代表スポットの theta 方向プロファイル比較
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor("#181818")
    if profile_data and "prof_before" in profile_data:
        rel_x = profile_data["rel_pixels"]
        pb = profile_data["prof_before"]
        pa = profile_data["prof_after"]
        fwhm_b = profile_data.get("fwhm_before", 0.0)
        fwhm_a = profile_data.get("fwhm_after", 0.0)
        s_id = profile_data.get("spot_id", 1)
        th_deg = profile_data.get("theta_deg", 0.0)

        # 両端（窓の左右端 4px）の中央値を局所ベースライン（背景オフセット）として推定し減算
        # これにより広域背景残差や近接スポットの足元による「不自然な tail の浮き上がり」を排除
        edge_n = max(len(pb) // 8, 3)
        pb_base = float(np.median(np.concatenate([pb[:edge_n], pb[-edge_n:]])))
        pa_base = float(np.median(np.concatenate([pa[:edge_n], pa[-edge_n:]])))

        pb_clean = np.maximum(0.0, pb - pb_base)
        pa_clean = np.maximum(0.0, pa - pa_base)

        pb_norm = pb_clean / max(pb_clean.max(), 1e-5)
        pa_norm = pa_clean / max(pa_clean.max(), 1e-5)

        ax5.plot(rel_x, pb_norm, "--", color="#3498db", linewidth=2.0, label=f"Before Deblur (FWHM: {fwhm_b:.2f} px)")
        ax5.plot(rel_x, pa_norm, "-", color="#e74c3c", linewidth=2.5, label=f"After Deblur (FWHM: {fwhm_a:.2f} px, Sharpened)")
        ax5.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2, label="Half-Maximum (50%)")

        reduction = (1.0 - (fwhm_a / fwhm_b)) * 100.0 if fwhm_b > 0 else 0.0
        
        flux_info = ""
        if physics_report:
            flux_info = f" | Total Flux Ratio: {physics_report.mean_flux_ratio:.3f}"

        ax5.set_title(
            f"5. Representative Spot Profile Comparison (Spot #{s_id} at θ={th_deg:.1f}°) | FWHM Reduction: {reduction:.1f}%{flux_info}",
            color="#ffffff",
            fontsize=12,
            pad=6,
        )
    else:
        ax5.text(0.5, 0.5, "No spot profile available", color="#aaaaaa", ha="center", va="center")
        ax5.set_title("5. Spot Profile Comparison", color="#ffffff", fontsize=12)

    ax5.set_xlabel("Relative Azimuth Offset Δθ [pixels]", color="#e0e0e0", fontsize=10)
    ax5.set_ylabel("Normalized Intensity", color="#e0e0e0", fontsize=10)
    ax5.grid(True, linestyle="--", alpha=0.3, color="#666666")
    ax5.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#e0e0e0", fontsize=10)
    ax5.tick_params(colors="#888888")

    if save_path is not None:
        p = Path(save_path)
        safe_savefig(fig, p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info("診断プロット図を保存しました: %s", p)

    return fig


def plot_batch_summary_figure(
    results_list: List[Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """バッチ処理結果の一覧比較図 (batch_summary_plot.png) を生成します.

    Args:
        results_list: 各画像のサマリー辞書のリスト
        output_path: 保存先パス

    Returns:
        output_path: 保存されたパス
    """
    n_images = len(results_list)
    if n_images == 0:
        return Path(output_path)

    # 1行に各画像のサムネイルと指標を表示
    fig, axes = plt.subplots(
        n_images, 3,
        figsize=(15, 3.5 * n_images),
        facecolor="#1e1e1e",
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.5]},
    )
    if n_images == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"AstroLaue Batch Processing Summary Report (Total {n_images} Images)",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        y=0.99,
    )

    for i, res in enumerate(results_list):
        ax_raw = axes[i, 0]
        ax_rest = axes[i, 1]
        ax_prof = axes[i, 2]

        ax_raw.set_facecolor("#121212")
        ax_rest.set_facecolor("#121212")
        ax_prof.set_facecolor("#181818")

        raw_name = res.get("image_name", f"Image #{i+1}")
        # CJK文字の警告防止 (ASCII安全表記)
        safe_name = raw_name.replace("画像", "Image_")
        name = safe_name
        status = res.get("status", "UNKNOWN")


        if status == "SUCCESS":
            crop_img = res.get("cropped_image")
            rest_img = res.get("restored_image")
            prof_data = res.get("profile_data", {})
            blur_l = res.get("blur_length", 0.0)
            red_pct = res.get("avg_fwhm_reduction", 0.0)
            n_spots = res.get("num_spots", 0)

            if crop_img is not None:
                if len(crop_img.shape) == 3:
                    ax_raw.imshow(crop_img[..., ::-1])
                else:
                    ax_raw.imshow(crop_img, cmap="gray")
            ax_raw.set_title(f"{name} (Raw)", color="#ffffff", fontsize=10)
            ax_raw.axis("off")

            if rest_img is not None:
                ax_rest.imshow(rest_img, cmap="inferno")
            ax_rest.set_title(f"Restored (Spots: {n_spots}, L={blur_l:.1f}px)", color="#ffffff", fontsize=10)
            ax_rest.axis("off")

            if prof_data and "prof_before" in prof_data:
                rx = prof_data["rel_pixels"]
                pb = prof_data["prof_before"]
                pa = prof_data["prof_after"]
                pb_n = (pb - pb.min()) / max(pb.max() - pb.min(), 1e-5)
                pa_n = (pa - pa.min()) / max(pa.max() - pa.min(), 1e-5)

                ax_prof.plot(rx, pb_n, "--", color="#3498db", label="Raw")
                ax_prof.plot(rx, pa_n, "-", color="#e74c3c", label=f"Restored (-{red_pct:.1f}%)")
                ax_prof.axhline(0.5, color="#888888", linestyle=":", alpha=0.6)
                ax_prof.set_title(f"FWHM Sharpening: {red_pct:.1f}% Reduction", color="#00ffcc", fontsize=10)
                ax_prof.legend(loc="upper right", facecolor="#2b2b2b", edgecolor="none", labelcolor="#e0e0e0", fontsize=8)
            else:
                ax_prof.text(0.5, 0.5, "No Profile", color="#aaaaaa", ha="center", va="center")
            ax_prof.tick_params(colors="#888888")
        else:
            ax_raw.text(0.5, 0.5, "FAILED", color="#ff4444", ha="center", va="center", fontsize=14)
            ax_raw.set_title(name, color="#ff4444")
            ax_rest.axis("off")
            ax_prof.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_p = Path(output_path)
    safe_savefig(fig, out_p, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("バッチ比較一覧図を保存しました: %s", out_p)
    return out_p


def plot_comparison_figure(
    cropped_orig: np.ndarray,
    restored_cart: np.ndarray,
    center_norm: Tuple[float, float],
    r_outer: float,
    spots: List[SpotResult],
    save_path: str | Path,
    blur_length: float = 0.0,
    physics_report: Any | None = None,
) -> Path:
    """元画像 (処理前) と復元後画像を横並びで直接比較する高解像度画像を生成・保存します.

    回折円盤の周囲の不要な余白を完全に排除し、処理前後のスポット先鋭化を
    客観的にひと目で検証できるようにレイアウトします。

    Args:
        cropped_orig: 元の切り出しROI画像 (BGRまたはグレースケール)
        restored_cart: 復元直交画像 (1024x1024等)
        center_norm: 復元画像上の中心座標
        r_outer: 外周円盤半径
        spots: 検出スポットリスト
        save_path: 保存先パス (comparison_before_after.png)
        blur_length: 推定ブラー長
        physics_report: 物理妥当性検証レポート

    Returns:
        saved_path: 保存されたファイルパス
    """
    import cv2
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. 復元画像のタイトクロップ
    h_r, w_r = restored_cart.shape[:2]
    crop_r = int(round(r_outer * 1.06))
    xc_r, yc_r = int(round(center_norm[0])), int(round(center_norm[1]))

    x1_r = max(0, xc_r - crop_r)
    y1_r = max(0, yc_r - crop_r)
    x2_r = min(w_r, xc_r + crop_r)
    y2_r = min(h_r, yc_r + crop_r)
    side_r = min(x2_r - x1_r, y2_r - y1_r)

    rest_tight = restored_cart[y1_r : y1_r + side_r, x1_r : x1_r + side_r].copy()

    # 2. 元画像のタイトクロップ (同じ視野になるよう中心基準)
    if len(cropped_orig.shape) == 3:
        orig_gray = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = cropped_orig.copy()

    h_o, w_o = orig_gray.shape[:2]
    # 元画像の中心 (概ね中央)
    xc_o, yc_o = w_o // 2, h_o // 2
    x1_o = max(0, xc_o - crop_r)
    y1_o = max(0, yc_o - crop_r)
    x2_o = min(w_o, xc_o + crop_r)
    y2_o = min(h_o, yc_o + crop_r)
    side_o = min(x2_o - x1_o, y2_o - y1_o)
    if side_o > 10:
        orig_tight = orig_gray[y1_o : y1_o + side_o, x1_o : x1_o + side_o].copy()
    else:
        orig_tight = orig_gray.copy()

    # サイズを統一
    target_side = max(side_r, orig_tight.shape[0])
    if rest_tight.shape[0] != target_side:
        rest_tight = cv2.resize(rest_tight, (target_side, target_side), interpolation=cv2.INTER_CUBIC)
    if orig_tight.shape[0] != target_side:
        orig_tight = cv2.resize(orig_tight, (target_side, target_side), interpolation=cv2.INTER_CUBIC)

    # ダイナミックレンジ調整 (Modified asinh ストレッチで微弱ピークを鮮明可視化)
    p99_orig = np.percentile(orig_tight, 99.8) if np.max(orig_tight) > 0 else 1.0
    orig_disp = np.clip(orig_tight.astype(np.float32) / max(p99_orig, 1e-4), 0.0, 1.0)
    orig_disp = np.power(orig_disp, 0.75)

    rest_f = np.maximum(rest_tight.astype(np.float32), 0.0)
    p99_rest = float(np.percentile(rest_f, 99.8)) if np.max(rest_f) > 0 else 1.0
    pos_rest = rest_f[rest_f > 0]
    if len(pos_rest) > 100:
        med_r = float(np.median(pos_rest))
        mad_r = float(np.median(np.abs(pos_rest - med_r)))
        beta_r = max(3.0 * 1.4826 * mad_r, 0.03 * p99_rest)
    else:
        beta_r = 0.03 * p99_rest

    a_r = float(np.clip(beta_r / max(p99_rest, 1e-6), 1e-4, 1.0))
    stretch_r = AsinhStretch(a=a_r)
    rest_norm_linear = np.clip(rest_f / max(p99_rest, 1e-4), 0.0, 1.0)
    rest_disp = np.clip(stretch_r(rest_norm_linear), 0.0, 1.0)

    # メトリクスサマリー
    reductions = [s.fwhm_reduction_ratio for s in spots if s.fwhm_before > 0]
    avg_red = sum(reductions) / len(reductions) if reductions else 0.0
    flux = physics_report.mean_flux_ratio if physics_report else 1.0

    # 描画
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.5), facecolor="#1a1a1a")

    # 左: 元画像
    ax1.imshow(orig_disp, cmap="inferno", origin="upper")
    ax1.set_title("[Before] Original Captured Image (Raw ROI)\nAzimuthal scanning motion blur present", color="#ffffff", fontsize=12, pad=12)
    ax1.axis("off")

    # 右: 復元後
    ax2.imshow(rest_disp, cmap="inferno", origin="upper")
    ax2.set_title("[After] AstroLaue Restored Image (Sharpened)\nDeblurred via Richardson-Lucy deconvolution", color="#00ffcc", fontsize=12, pad=12)
    ax2.axis("off")

    # スポット位置のプロット (クロップ座標系へシフト)
    scale_factor = target_side / float(side_r)
    for s in spots[:50]:  # 上位50個
        # 元の座標 (x, y) からクロップオフセットを引く
        sx_c = (s.x - x1_r) * scale_factor
        sy_c = (s.y - y1_r) * scale_factor
        if 0 <= sx_c < target_side and 0 <= sy_c < target_side:
            circle = patches.Circle((sx_c, sy_c), radius=max(s.sigma * 1.5 * scale_factor, 6.0), edgecolor="#00ffcc", facecolor="none", linewidth=1.2, linestyle="--")
            ax2.add_patch(circle)
            ax2.text(sx_c + 7, sy_c - 7, f"{s.spot_id}", color="#00ffcc", fontsize=8, fontweight="bold")

    # 全体ヘッダータイトル
    header_text = (
        f"AstroLaue Restoration Comparison  |  Blur Length L_theta: {blur_length:.1f} px  |  "
        f"Avg FWHM Reduction: {avg_red:.1f}%  |  Spots: {len(spots)}  |  Flux Ratio: {flux:.3f}"
    )
    fig.suptitle(header_text, color="#ffffff", fontsize=13, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    safe_savefig(fig, out_p, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("ビフォーアフター検証比較画像を保存しました: %s", out_p)
    return out_p

