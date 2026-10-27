"""モジュール: 露光時間最適化＆ピーク飽和解析 (astrolaue.exposure).

Rigaku Astray / sLaue 検出器のカウント飽和（サチュレーション）を判定し、
回転ブラーの影響を受けない半径 (r) 方向の未飽和裾野プロファイルから
真のピーク最大強度 (I_true_max) を逆算・外挿します。
デコンボリューション精度を最大化するための推奨測定時間 (上限80%目標) を瞬時に算定します。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.ndimage import maximum_filter

from astrolaue.io import ExposureMetadata, SlaueRawData, get_exposure_time_from_asts

logger = logging.getLogger(__name__)


class SaturationStatus(str, Enum):
    """露光・飽和ステータス."""

    SATURATED = "SATURATED"
    """飽和検知 (上限到達・頭打ち). 短縮を推奨."""

    SLIGHTLY_HIGH = "SLIGHTLY_HIGH"
    """上限近傍 (飽和寸前: 88〜95%). 微小短縮を推奨."""

    OPTIMAL = "OPTIMAL"
    """適正露光 (目標80%近傍: 72〜88%). 変更不要."""

    UNDER_EXPOSED = "UNDER_EXPOSED"
    """露光不足 (72%未満). SN比向上のための延長を推奨."""


@dataclass
class ExposureOptimizationResult:
    """露光時間最適化＆ピーク解析結果."""

    status: SaturationStatus
    """露光ステータス."""

    observed_max: float
    """観測された最大測定強度 [counts]."""

    estimated_true_max: float
    """未飽和裾野から推定された真のピーク最大強度 [counts]."""

    background: float
    """背景ノイズ・散乱光中央値 [counts]."""

    saturation_limit: float
    """検出器飽和上限値 [counts]."""

    target_max: float
    """目標最大強度 (上限の80%) [counts]."""

    recommended_ratio: float
    """推奨スケーリング比率 (目標時間 / 現行時間). 1.0 未満は短縮、1.0 超は延長."""

    peak_r_mm: float
    """最大ピークの動径位置 [mm]."""

    peak_theta_deg: float
    """最大ピークの方位角 [deg]."""

    current_seconds: Optional[float] = None
    """現行の総照射時間 [秒] (.ASTS から取得、未取得時は None)."""

    single_exposure_seconds: Optional[float] = None
    """現行の1回あたり測定時間 [秒] (例: 60.0秒)."""

    recommended_seconds: Optional[float] = None
    """推奨の総照射時間 [秒] (現行測定時間が既知の場合)."""

    recommended_single_seconds: Optional[float] = None
    """推奨の1回あたり測定時間 [秒] (案1: 積算回数を維持して1回時間を調整)."""

    num_accumulations: int = 1
    """現行の積算回数."""

    radial_integral: float = 0.0
    """選定された支配ピークの半径方向積分強度 [counts]."""

    num_candidate_peaks: int = 1
    """画面内で検出された飽和・強スポット候補総数."""

    def format_section(self) -> str:
        """アライメントセクションと完全に独立したコンソール表示文字列を生成します."""
        lines = []
        lines.append("=" * 70)
        lines.append("【⏱️ 露光時間最適化＆ピーク解析】")

        ratio_to_sat = (self.observed_max / self.saturation_limit) * 100.0
        if self.status == SaturationStatus.SATURATED:
            status_badge = "【⚠️ 飽和検知 / 上限到達】"
            advice_badge = "短縮設定を推奨"
        elif self.status == SaturationStatus.SLIGHTLY_HIGH:
            status_badge = "【⚠️ やや高め / 飽和寸前】"
            advice_badge = "微小短縮を推奨"
        elif self.status == SaturationStatus.OPTIMAL:
            status_badge = "【✅ 最適露光 (目標範囲内)】"
            advice_badge = "現行設定の維持を推奨"
        else:
            status_badge = "【ℹ️ 露光不足】"
            advice_badge = "SN比向上のため延長を推奨"

        lines.append(f"  * 露光ステータス      : {status_badge}")
        cand_str = f" (候補 {self.num_candidate_peaks} 個から半径積分最大ピークを選定)" if self.num_candidate_peaks > 1 else ""
        lines.append(
            f"  * 最強ピーク位置      : r = {self.peak_r_mm:.2f} mm, θ = {self.peak_theta_deg:.2f}°{cand_str}"
        )
        lines.append(
            f"  * 最大測定強度 (Max)  : {self.observed_max:,.0f} counts "
            f"(飽和上限 {self.saturation_limit:,.0f} の {ratio_to_sat:.1f}%)"
        )

        if self.estimated_true_max > self.observed_max:
            true_ratio = (self.estimated_true_max / self.saturation_limit) * 100.0
            lines.append(
                f"  * 裾野推定 真のピーク : {self.estimated_true_max:,.0f} counts "
                f"(上限比 {true_ratio:.1f}%) ★未飽和裾野より外挿"
            )

        lines.append(
            f"  * 目標最大強度 (80%)  : {self.target_max:,.0f} counts "
            f"(デコンボリューション精度を最大化する安全最適値)"
        )

        pct = self.recommended_ratio * 100.0
        if self.current_seconds is not None:
            if self.single_exposure_seconds is not None:
                lines.append(
                    f"  * 現行測定条件 (ASTS) : 測定時間 {self.single_exposure_seconds:.1f} 秒 "
                    f"× {self.num_accumulations} 回積算 (総照射 {self.current_seconds:.1f} 秒)"
                )
            else:
                acc_str = f" ({self.num_accumulations}回積算)" if self.num_accumulations > 1 else ""
                lines.append(f"  * 現行測定時間 (ASTS) : {self.current_seconds:.1f} 秒{acc_str}")

            if self.recommended_single_seconds is not None and self.recommended_seconds is not None:
                lines.append("  * 【推奨設定 (案1)】  :")
                lines.append(
                    f"      >>> 測定時間の設定 : {self.recommended_single_seconds:.1f} 秒 "
                    f"(積算回数は {self.num_accumulations} 回のまま維持、総照射 {self.recommended_seconds:.1f} 秒)"
                )
                lines.append(
                    f"      >>> 設定推奨比率   : 現行測定時間の {pct:.0f}% に{advice_badge}"
                )
                if self.recommended_seconds > 600.0:
                    lines.append(
                        "      ※ 注: 総照射時間が10分を超えるため、まずは段階的な延長 (例: 測定時間 120秒〜180秒) での様子見を推奨します"
                    )
            elif self.recommended_seconds is not None:
                lines.append("  * 【推奨測定時間】    :")
                lines.append(
                    f"      >>> 目標露光時間   : {self.recommended_seconds:.1f} 秒 "
                    f"(現行の {pct:.0f}% に{advice_badge})"
                )
        else:
            lines.append("  * 【推奨設定 (案1)】  :")
            lines.append(
                f"      >>> 測定時間の設定 : 現行の「測定時間」を {pct:.0f}% に設定変更を推奨 "
                f"(積算回数は維持、{advice_badge})"
            )

        if self.status == SaturationStatus.SATURATED:
            lines.append("  * 改善効果            : 1回ごとの物理飽和とピーク頭打ちを防ぎ、デブラー先鋭化を極限まで発揮")
        elif self.status == SaturationStatus.SLIGHTLY_HIGH:
            lines.append("  * 改善効果            : 飽和マージンを確保し、測定強度の線形性とデブラー復元精度を担保")
        elif self.status == SaturationStatus.OPTIMAL:
            lines.append("  * 判定                : 現在の照射条件で最高のデブラー復元精度が得られています")
        elif self.status == SaturationStatus.UNDER_EXPOSED:
            lines.append("  * 改善効果            : 1回あたりの照射時間を延長することで微弱ピークのS/N比が向上し、弱い回折斑点の検出感度が向上")

        lines.append("=" * 70)
        return "\n".join(lines)


def analyze_exposure_and_optimize(
    raw_data: Union[SlaueRawData, np.ndarray],
    saturation_limit: float = 7.72e6,
    target_ratio: float = 0.80,
    source_path: Optional[Union[str, Path]] = None,
) -> ExposureOptimizationResult:
    """極座標生データからピーク飽和を解析し、最適測定時間を算定します.

    ※注: 飽和マスク等のデータ欠損・黒塗り処理は一切適用しません。
    半径方向の未飽和スロープの曲率から真の最大強度を純粋な外挿演算で逆算します。

    Args:
        raw_data: SlaueRawData オブジェクト または 極座標2D配列 (カウント値)
        saturation_limit: 検出器の飽和上限値 [counts] (デフォルト: 7,724,134 counts 付近)
        target_ratio: 目標とする飽和比率 (デフォルト: 0.80 = 上限の80%)
        source_path: CSV等のファイルパス (同一フォルダの .ASTS を自動探索するため)

    Returns:
        ExposureOptimizationResult オブジェクト
    """
    if isinstance(raw_data, SlaueRawData):
        p_data = raw_data.polar_data
        radii_mm = raw_data.radii_mm
        thetas_deg = raw_data.thetas_deg
        csv_file = raw_data.file_path or (Path(source_path) if source_path else None)
    else:
        p_data = np.asarray(raw_data, dtype=np.float32)
        radii_mm = np.linspace(5.0, 40.0, p_data.shape[0], dtype=np.float32)
        thetas_deg = np.linspace(0.0, 360.0, p_data.shape[1], endpoint=False, dtype=np.float32)
        csv_file = Path(source_path) if source_path else None

    bg_val = float(np.median(p_data))
    target_max = float(saturation_limit * target_ratio)

    # 1. 強いスポット候補のマルチピックアップ (maximum_filter による局所極大値検出)
    # 周囲 (7x15) の中で極大であり、かつ有意に高いピークを全件抽出
    filt = maximum_filter(p_data, size=(7, 15))
    rel_thresh = bg_val + 0.50 * (float(p_data.max()) - bg_val)
    peak_thresh = min(saturation_limit * 0.70, rel_thresh)
    cand_mask = (p_data == filt) & (p_data >= peak_thresh)
    cand_r_idx, cand_th_idx = np.where(cand_mask)

    if len(cand_r_idx) == 0:
        # フォールバック: 全体最大値の1点
        r_max_idx, th_max_idx = np.unravel_index(np.argmax(p_data), p_data.shape)
        candidates = [(r_max_idx, th_max_idx)]
    else:
        candidates = list(zip(cand_r_idx, cand_th_idx))

    # 2. 【超高速スクリーニング】 半径方向の積分値 (Radial Integral) による真の最強ピーク選定
    # ピークトップが飽和で平坦化していても、半径方向の積分値 (面積) は真の総エネルギーに直接比例します
    w_r = 8
    best_cand = candidates[0]
    best_radial_integral = -1.0

    for r_i, th_i in candidates:
        r_slice = np.arange(max(0, r_i - w_r), min(p_data.shape[0], r_i + w_r + 1))
        prof_r = p_data[r_slice, th_i]
        sig_r = np.maximum(prof_r - bg_val, 0.0)
        r_int = float(np.sum(sig_r))
        if r_int > best_radial_integral:
            best_radial_integral = r_int
            best_cand = (r_i, th_i)

    # 選定された「真の最強ピーク」の座標と観測値
    sel_r_idx, sel_th_idx = best_cand
    obs_max = float(p_data[sel_r_idx, sel_th_idx])
    peak_r = float(radii_mm[sel_r_idx])
    peak_th = float(thetas_deg[sel_th_idx])

    # 3. 飽和判定
    is_saturated = obs_max >= (saturation_limit * 0.95)
    if not is_saturated and obs_max >= (saturation_limit * 0.90):
        # プラトーチェック (隣接画素のいずれかと等値・平坦か)
        r_slice_check = p_data[max(0, sel_r_idx - 1) : min(p_data.shape[0], sel_r_idx + 2), sel_th_idx]
        if len(r_slice_check) >= 2 and np.min(np.abs(np.diff(r_slice_check))) < (obs_max * 0.015):
            is_saturated = True

    # 4. 選定された最強スポット1点に対してのみ、真のピーク最大強度を逆算
    estimated_true_max = obs_max
    if is_saturated:
        try:
            r_indices = np.arange(max(0, sel_r_idx - w_r), min(p_data.shape[0], sel_r_idx + w_r + 1))
            prof_r = p_data[r_indices, sel_th_idx]
            dr = (r_indices - sel_r_idx).astype(np.float64)

            sig = prof_r - bg_val
            peak_sig = obs_max - bg_val

            # (A) ガウス分布の積分関係 (Area = sqrt(2*pi)*sigma*S_peak ≒ 5.0 * S_peak) による面積逆算
            # スポット幅 sigma ≒ 2.0 px の標準的なラウエ光学系形状
            approx_sigma = 2.0
            area_pred_max = bg_val + (best_radial_integral / (np.sqrt(2.0 * np.pi) * approx_sigma))

            # (B) 未飽和スロープ (裾野) の対数放物線回帰
            valid = (sig > 0.15 * peak_sig) & (prof_r < saturation_limit * 0.88)
            fit_pred_max = obs_max

            if np.sum(valid) >= 3:
                y = np.log(np.maximum(sig[valid], 1.0))
                x = dr[valid]
                poly = np.polyfit(x, y, 2)
                a, b, c = poly
                if a < -1e-5:  # 上に凸の放物線
                    dr_peak = -b / (2.0 * a)
                    if abs(dr_peak) < 2.5:
                        exponent = c - (b**2) / (4.0 * a)
                        # 数値オーバーフローおよび物理サニティガード
                        if np.isfinite(exponent) and exponent < 20.0:
                            pred_sig = float(np.exp(exponent))
                            fit_pred_max = bg_val + pred_sig

            # 裾野回帰値と面積逆算値から、より大きく安全な真の最大強度を採用
            estimated_true_max = max(obs_max, fit_pred_max, area_pred_max)
        except Exception as e_fit:
            logger.debug("未飽和裾野フィッティング中の例外: %s", e_fit)
            estimated_true_max = obs_max

    # 5. ステータス決定
    if is_saturated or (estimated_true_max >= saturation_limit * 0.98):
        status = SaturationStatus.SATURATED
    elif obs_max >= (saturation_limit * 0.88):
        status = SaturationStatus.SLIGHTLY_HIGH
    elif obs_max >= (saturation_limit * 0.72):
        status = SaturationStatus.OPTIMAL
    else:
        status = SaturationStatus.UNDER_EXPOSED

    # 5. 推奨スケーリング比率の算出
    # 目標強度 (上限80%) / 推定真の最大強度
    rec_ratio = float(target_max / max(estimated_true_max, 1.0))

    # 6. 現行測定時間 (.ASTS) の取得と推奨秒数の計算 (案1: 1回あたりの測定時間を調整)
    curr_seconds: Optional[float] = None
    curr_single: Optional[float] = None
    rec_seconds: Optional[float] = None
    rec_single: Optional[float] = None
    num_acc = 1

    if csv_file is not None:
        meta = get_exposure_time_from_asts(csv_file)
        if meta is not None:
            curr_seconds = meta.total_seconds
            curr_single = meta.single_exposure_seconds
            num_acc = meta.num_accumulations
            rec_seconds = max(1.0, round(curr_seconds * rec_ratio, 1))
            rec_single = max(0.5, round(curr_single * rec_ratio, 1))

    return ExposureOptimizationResult(
        status=status,
        observed_max=obs_max,
        estimated_true_max=estimated_true_max,
        background=bg_val,
        saturation_limit=saturation_limit,
        target_max=target_max,
        current_seconds=curr_seconds,
        single_exposure_seconds=curr_single,
        recommended_seconds=rec_seconds,
        recommended_single_seconds=rec_single,
        recommended_ratio=rec_ratio,
        peak_r_mm=peak_r,
        peak_theta_deg=peak_th,
        num_accumulations=num_acc,
        radial_integral=best_radial_integral,
        num_candidate_peaks=len(candidates),
    )
