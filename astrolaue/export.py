"""課題 2: 提出用画像フォーマット整備 (Publication-Ready Image Export).

論文・学会発表・報告書提出向けの高品質な画像出力:
1. 透過PNG (RGBA): ダイレクトビーム穴内部および外周円盤外部を完全透過 (Alpha=0)
2. クリーンな学術用グレースケール / 疑似カラー / 16-bit PNG / ダイナミックレンジ自動補正
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Literal

import cv2
import time
import numpy as np
from astropy.visualization import AsinhStretch

logger = logging.getLogger(__name__)


def safe_imwrite(path: str | Path, img: np.ndarray) -> bool:
    """Windows/Linux 対応の安全な画像保存関数 (Unicodeパス対応 & ロック検知・フォールバック)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        ext = p.suffix.lower() or ".png"
        success, encoded = cv2.imencode(ext, img)
        if success:
            p.write_bytes(encoded.tobytes())
            return True
    except PermissionError as pe:
        logger.warning("ファイルが別アプリケーションで開かれているため上書きできませんでした (%s): %s", p, pe)
        try:
            alt_path = p.with_name(f"{p.stem}_{time.strftime('%H%M%S')}{p.suffix}")
            alt_path.write_bytes(encoded.tobytes())
            logger.info("代替ファイル名として保存しました: %s", alt_path)
            return True
        except Exception:
            pass
    except Exception as ex:
        logger.debug("imencode 書き込み失敗、標準 imwrite を試行: %s", ex)

    return bool(cv2.imwrite(str(p), img))


def create_circular_mask(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    r_inner: float,
    r_outer: float,
    feather_pix: float = 1.5,
    mask_inner_hole: bool = False,
) -> np.ndarray:
    """有効回折領域を抽出し、境界をアンチエイリアス処理したアルファマスク (0.0 - 1.0) を生成します.

    Args:
        shape: 画像サイズ (H, W)
        center: 中心座標 (xc, yc)
        r_inner: 内周（ビームストッパー）半径
        r_outer: 外周円盤半径
        feather_pix: エッジのぼかし（アンチエイリアシング）幅
        mask_inner_hole: Trueの場合、中心穴も透過(0.0)にする。Falseの場合、中心は不透明で外周外側のみ透過。

    Returns:
        alpha_mask: float32 配列 (0.0: 完全透過, 1.0: 完全不透過)
    """
    h, w = shape
    xc, yc = center

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dists = np.hypot(x_indices - xc, y_indices - yc)

    if feather_pix > 0:
        alpha_outer = np.clip(((r_outer + feather_pix) - dists) / (2 * feather_pix), 0.0, 1.0)
        if mask_inner_hole:
            alpha_inner = np.clip((dists - (r_inner - feather_pix)) / (2 * feather_pix), 0.0, 1.0)
            alpha = alpha_inner * alpha_outer
        else:
            alpha = alpha_outer
    else:
        if mask_inner_hole:
            alpha = ((dists >= r_inner) & (dists <= r_outer)).astype(np.float32)
        else:
            alpha = (dists <= r_outer).astype(np.float32)

    return alpha.astype(np.float32)


def export_publication_rgba(
    image: np.ndarray,
    center: Tuple[float, float],
    r_inner: float,
    r_outer: float,
    output_path: str | Path,
    colormap: Optional[str] = None,
    invert_grayscale: bool = False,
    gamma: float = 1.0,
    percentile_clip: float = 100.0,
    export_16bit: bool = True,
    tight_crop: bool = False,
    crop_margin_ratio: float = 1.06,
    mask_inner_hole: bool = False,
    noise_gate_ratio: float = 0.01,
    use_asinh: bool = True,
    asinh_beta: Optional[float] = None,
) -> Path:
    """論文・報告書向けに背景透過PNG (RGBA) を出力します.

    16-bit PNG は定量的物理解析のために 100% 厳密な線形（リニア）強度を保持します。
    8-bit 表示用PNG（restored_transparent.png）には、天体画像標準の Modified asinh
    （逆双曲線正弦）トーンマッピングを適用することで、超強ピークの飽和を防ぎつつ、
    元画像に存在していた微弱ピークのコントラストを自然に引き上げて肉眼での視認性を飛躍的に高めます。
    中心の基準白丸は不透明白色で保持し、外周円盤の外側の余白のみを透過処理します。

    Args:
        image: 復元直交画像 (float32 [0.0, 1.0] または 2次元配列)
        center: 画像中心座標 (xc, yc)
        r_inner: ビームストッパー半径 (中心白色円境界)
        r_outer: 外周円盤半径 (透過境界)
        output_path: 保存先ファイルパス (例: restored_transparent.png)
        colormap: カラーマップ名 (None: グレースケール, 'inferno', 'viridis' 等)
        invert_grayscale: Trueの場合、白円盤に黒スポット (X線回折標準反転)
        gamma: ガンマ補正値 (デフォルト: 1.0 = リニア、暗部ノイズ増幅を防止)
        percentile_clip: 上限輝度クリップのパーセンタイル (デフォルト: 100.0 = 最大値基準)
        export_16bit: Trueの場合、16-bit深度で保存 (線形強度厳密保持)
        tight_crop: Trueの場合、回折円盤周囲の広大な余白をカットしてジャストサイズで保存
        crop_margin_ratio: 外周円盤半径に対するマージン倍率 (デフォルト: 1.06)
        mask_inner_hole: Trueの場合、中心穴も透過(0.0)にする (デフォルト: False=中心白色)
        noise_gate_ratio: 最大値に対する微細ノイズカット閾値比率 (デフォルト: 0.01 = 1%未満カット)
        use_asinh: 8-bit 出力時に天体画像標準 Modified asinh ストレッチを適用するか (デフォルト: True)
        asinh_beta: asinh ストレッチの変曲点パラメータ (None で背景MADから自動決定)

    Returns:
        saved_path: 保存された画像パス
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    img_f = np.maximum(image.astype(np.float32), 0.0)

    # 有効円盤内ピクセルをサンプリングしてパーセンタイル決定
    h, w = img_f.shape[:2]
    alpha = create_circular_mask((h, w), center, r_inner, r_outer, mask_inner_hole=mask_inner_hole)
    valid_pixels = img_f[alpha > 0.5]

    if len(valid_pixels) > 0 and np.max(valid_pixels) > 0:
        if percentile_clip < 100.0:
            high_val = float(np.percentile(valid_pixels, percentile_clip))
        else:
            high_val = float(np.max(valid_pixels))

        if high_val <= 0:
            high_val = float(np.max(valid_pixels))

        # 1. 物理解析・16-bit用の厳密な線形（リニア）正規化
        norm_linear = np.clip(img_f / high_val, 0.0, 1.0)

        # 2. 8-bit表示・可視化用のストレッチ (astropy.visualization.AsinhStretch 公式クラス活用)
        if use_asinh:
            eff_beta = asinh_beta
            if eff_beta is None:
                pos = valid_pixels[valid_pixels > 0]
                if len(pos) > 100:
                    med = float(np.median(pos))
                    mad = float(np.median(np.abs(pos - med)))
                    sigma_mad = 1.4826 * mad
                    eff_beta = max(3.0 * sigma_mad, 0.03 * high_val)
                else:
                    eff_beta = 0.03 * high_val

            # a = eff_beta / high_val (正規化空間での変曲点)
            a_val = float(np.clip(eff_beta / max(high_val, 1e-6), 1e-4, 1.0))
            stretch = AsinhStretch(a=a_val)
            norm_img = np.clip(stretch(norm_linear), 0.0, 1.0)
        else:
            norm_img = norm_linear

        # 背景の微細ノイズゲート (最大値の 1% 未満の浮動小数点残差を滑らかに 0 に落とす)
        if noise_gate_ratio > 0.0:
            norm_img = np.where(
                norm_img < noise_gate_ratio,
                norm_img * (norm_img / max(noise_gate_ratio, 1e-4)) ** 2,
                norm_img,
            )
    else:
        norm_linear = img_f
        norm_img = img_f

    # ガンマ補正 (明示的に指定された場合のみ適用)
    if gamma != 1.0:
        norm_img = np.power(norm_img, gamma)

    # 中心白丸 (ダイレクトビーム基準円) を自然な白色 (1.0) として描画 (中心穴を開けない場合)
    inner_mask = None
    if not mask_inner_hole:
        y_idx, x_idx = np.indices((h, w), dtype=np.float32)
        dists = np.hypot(x_idx - center[0], y_idx - center[1])
        inner_mask = (dists <= r_inner)
        norm_img[inner_mask] = 1.0
        norm_linear[inner_mask] = 1.0
        alpha[inner_mask] = 1.0

    # カラーまたはグレースケール生成 (8-bit)
    if colormap is not None:
        import matplotlib as mpl
        try:
            cmap_func = mpl.colormaps[colormap]
        except Exception:
            import matplotlib.pyplot as plt
            cmap_func = plt.get_cmap(colormap)
        rgba_f = cmap_func(norm_img)
        rgba_f[..., 3] *= alpha
        rgba_u8 = (np.clip(rgba_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgra = cv2.cvtColor(rgba_u8, cv2.COLOR_RGBA2BGRA)
    else:
        if invert_grayscale:
            intensity = 1.0 - norm_img
        else:
            intensity = norm_img

        intensity_u8 = (np.clip(intensity, 0.0, 1.0) * 255.0).astype(np.uint8)
        alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        bgra = cv2.merge([intensity_u8, intensity_u8, intensity_u8, alpha_u8])

    # 16-bit 版の配列準備 (100% 厳密線形リニア)
    bgra_16 = None
    if export_16bit:
        intensity_u16 = (np.clip(norm_linear, 0.0, 1.0) * 65535.0).astype(np.uint16)
        alpha_u16 = (np.clip(alpha, 0.0, 1.0) * 65535.0).astype(np.uint16)
        bgra_16 = cv2.merge([intensity_u16, intensity_u16, intensity_u16, alpha_u16])

    # 反転表示 (白背景・黒スポット) 配列準備
    inv_u8 = (np.clip(1.0 - norm_img, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgra_inv = cv2.merge([inv_u8, inv_u8, inv_u8, alpha_u8])

    # 余白除去 (タイトクロップ)
    if tight_crop and r_outer > 0:
        crop_r = int(round(r_outer * crop_margin_ratio))
        xc_i = int(round(center[0]))
        yc_i = int(round(center[1]))
        x1 = max(0, xc_i - crop_r)
        y1 = max(0, yc_i - crop_r)
        x2 = min(w, xc_i + crop_r)
        y2 = min(h, yc_i + crop_r)

        # 正方形化
        side = min(x2 - x1, y2 - y1)
        x2 = x1 + side
        y2 = y1 + side

        bgra = bgra[y1:y2, x1:x2].copy()
        if bgra_16 is not None:
            bgra_16 = bgra_16[y1:y2, x1:x2].copy()
        bgra_inv = bgra_inv[y1:y2, x1:x2].copy()
        logger.info("余白をカットしてタイトクロップしました: サイズ=%dx%d", side, side)

    # 書き出し
    safe_imwrite(out_p, bgra)
    logger.info("学術提出用透過PNGを保存しました: %s (サイズ=%dx%d)", out_p, bgra.shape[1], bgra.shape[0])

    if export_16bit and bgra_16 is not None:
        out_16 = out_p.with_name(f"{out_p.stem}_16bit.png")
        safe_imwrite(out_16, bgra_16)
        logger.info("16-bit透過PNGを保存しました: %s", out_16)

    out_inv = out_p.with_name(f"{out_p.stem}_inverted.png")
    safe_imwrite(out_inv, bgra_inv)

    return out_p


def export_contrast_optimized_image(
    image: np.ndarray,
    center: Tuple[float, float],
    r_inner: float,
    r_outer: float,
    output_path: str | Path,
    asinh_beta: float = 6.0,
    p_low_percentile: float = 2.0,
    p_high_percentile: float = 99.8,
    outer_bg_value: float = 0.95,
    export_pseudo_color: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """元画像形式 (中心白丸 + 有効円盤 + 余白背景) に準拠したコントラスト最適化画像を出力します.

    Modified asinh トーンマッピングとロバスト・パーセンタイルクリッピングにより、
    主反射ピークの白飛び（飽和）を防ぎながら、微弱スポットや晶帯軸の連続線を目視で最も鮮明に識別できるように最適化します。

    Args:
        image: 復元直交画像 (float32 [0.0, 1.0] または 2次元配列)
        center: 中心座標 (xc, yc)
        r_inner: 中心ビームストッパー白丸の半径
        r_outer: 有効回折円盤の半径
        output_path: 保存先ファイルパス (例: restored_contrast_optimized.png)
        asinh_beta: asinh の非線形圧縮強度 (大きいほど高輝度を強く圧縮し微弱光を強調)
        p_low_percentile: ブラックポイント（背景ノイズ足切り）のパーセンタイル
        p_high_percentile: ホワイトポイント（飽和保護）のパーセンタイル
        outer_bg_value: 円盤外側の背景輝度 (GUIウィンドウ同等の薄灰色 0.95)
        export_pseudo_color: Trueの場合、カラーマップ (Inferno) を適用した擬似カラー版も保存

    Returns:
        gray_path: グレースケール最適化画像の保存パス
        pseudo_path: 擬似カラー画像の保存パス (作成時のみ)
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    img_f = image.astype(np.float32)
    h, w = img_f.shape[:2]
    xc, yc = center

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dists = np.hypot(x_indices - xc, y_indices - yc)

    # 有効回折円盤内のピクセル
    disk_mask = (dists <= r_outer) & (dists >= r_inner)
    disk_pixels = img_f[disk_mask]

    if len(disk_pixels) > 50:
        p_low = float(np.percentile(disk_pixels, p_low_percentile))
        p_high = float(np.percentile(disk_pixels, p_high_percentile))
    else:
        p_low = 0.0
        p_high = float(img_f.max()) if img_f.max() > 0 else 1.0

    denom = max(p_high - p_low, 1e-4)
    # ノイズフロアクランプ
    scaled = np.clip((img_f - p_low) / denom, 0.0, None)

    # asinh ストレッチ
    beta = max(float(asinh_beta), 1.0)
    asinh_mapped = np.arcsinh(beta * scaled) / np.arcsinh(beta)
    asinh_mapped = np.clip(asinh_mapped, 0.0, 1.0)

    # 元画像形式の合成: 有効円盤内 = 最適化輝度、外側 = outer_bg_value (0.95)、中心白丸 = 1.0 (白)
    opt_canvas = asinh_mapped.copy()
    opt_canvas[dists > r_outer] = outer_bg_value
    opt_canvas[dists <= r_inner] = 1.0

    gray_u8 = (np.clip(opt_canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    safe_imwrite(out_p, gray_u8)
    logger.info("コントラスト最適化グレースケール画像を保存しました: %s", out_p)

    pseudo_path = None
    if export_pseudo_color:
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps["inferno"]
        except Exception:
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("inferno")
        # 円盤内のみカラーマップ適用
        rgba = cmap(asinh_mapped)[..., :3]  # RGB [0, 1]
        rgba[dists > r_outer] = outer_bg_value
        rgba[dists <= r_inner] = 1.0

        bgr_u8 = (np.clip(rgba[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
        pseudo_path = out_p.with_name(f"{out_p.stem}_pseudo.png")
        safe_imwrite(pseudo_path, bgr_u8)
        logger.info("コントラスト最適化擬似カラー画像を保存しました: %s", pseudo_path)

    return out_p, pseudo_path


def export_linear_image(
    image: np.ndarray,
    center: Tuple[float, float],
    r_inner: float,
    r_outer: float,
    output_path: str | Path,
    outer_bg_value: float = 0.95,
) -> Path:
    """元画像形式 (中心白丸 + 有効円盤 + 余白背景) に準拠した線形 (リニア) 復元画像を出力します.

    Args:
        image: 復元直交画像 (float32 [0.0, 1.0])
        center: 中心座標 (xc, yc)
        r_inner: 中心ビームストッパー白丸の半径
        r_outer: 有効回折円盤の半径
        output_path: 保存先ファイルパス (例: restored_linear.png)
        outer_bg_value: 余白背景の輝度

    Returns:
        output_path: 保存先ファイルパス
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    img_f = image.astype(np.float32)
    h, w = img_f.shape[:2]
    xc, yc = center

    y_indices, x_indices = np.indices((h, w), dtype=np.float32)
    dists = np.hypot(x_indices - xc, y_indices - yc)

    # 線形スケーリング (円盤内最大値を 1.0 に正規化)
    disk_mask = (dists <= r_outer) & (dists >= r_inner)
    d_max = float(img_f[disk_mask].max()) if np.any(disk_mask) else 1.0
    norm_linear = np.clip(img_f / max(d_max, 1e-4), 0.0, 1.0)

    canvas = norm_linear.copy()
    canvas[dists > r_outer] = outer_bg_value
    canvas[dists <= r_inner] = 1.0

    gray_u8 = (canvas * 255.0).astype(np.uint8)
    safe_imwrite(out_p, gray_u8)
    logger.info("元画像準拠線形復元画像を保存しました: %s", out_p)
    return out_p


def export_three_way_comparison(
    original_image: np.ndarray,
    linear_image: np.ndarray,
    optimized_image: np.ndarray,
    output_path: str | Path,
) -> Path:
    """[元画像 | 復元(線形) | 復元(コントラスト最適化)] の3並列比較図を生成・保存します.

    Args:
        original_image: クロップ/幾何規格化後の元画像
        linear_image: 線形復元画像
        optimized_image: コントラスト最適化画像
        output_path: 保存先ファイルパス (例: contrast_comparison.png)

    Returns:
        output_path: 保存先ファイルパス
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), facecolor="#1e1e1e")
    fig.suptitle("AstroLaue Image Comparison: Original vs. Linear Restored vs. Contrast Optimized", color="#ffffff", fontsize=15, fontweight="bold", y=0.98)

    titles = ["1. Original Raw Input", "2. Restored (Linear Deconvolution)", "3. Restored (Contrast Optimized HDR)"]
    subtitles = [
        "Blurred & Background Scattered",
        "Sharpened Bragg Spots (Physical Scale)",
        "asinh Tone Mapped (Maximized Weak Spot Visibility)",
    ]
    imgs = [original_image, linear_image, optimized_image]

    for i, ax in enumerate(axes):
        ax.set_facecolor("#121212")
        img_show = imgs[i]
        if len(img_show.shape) == 3:
            ax.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img_show, cmap="gray", vmin=0, vmax=255 if img_show.dtype == np.uint8 else 1.0)
        ax.set_title(f"{titles[i]}\n{subtitles[i]}", color="#ffffff", fontsize=11, pad=10)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(str(out_p), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("3並列コントラスト比較図を保存しました: %s", out_p)
    return out_p
