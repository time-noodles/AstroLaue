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
from typing import Optional, Tuple, Union

import numpy as np

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

    current_seconds: Optional[float]
    """現行の測定時間 [秒] (.ASTS から取得、未取得時は None)."""

    recommended_seconds: Optional[float]
    """推奨測定時間 [秒] (現行測定時間が既知の場合)."""

    recommended_ratio: float
    """推奨スケーリング比率 (目標時間 / 現行時間). 1.0 未満は短縮、1.0 超は延長."""

    peak_r_mm: float
    """最大ピークの動径位置 [mm]."""

    peak_theta_deg: float
    """最大ピークの方位角 [deg]."""

    num_accumulations: int = 1
    """現行の積算回数."""

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
        lines.append(
            f"  * 最大測定強度 (Max)  : {self.observed_max:,.0f} counts "
            f"(飽和上限 {self.saturation_limit:,.0f} の {ratio_to_sat:.1f}%)"
        )

        if self.status in [SaturationStatus.SATURATED, SaturationStatus.SLIGHTLY_HIGH]:
            true_ratio = (self.estimated_true_max / self.saturation_limit) * 100.0
            lines.append(
                f"  * 裾野推定 真のピーク : {self.estimated_true_max:,.0f} counts "
                f"(上限比 {true_ratio:.1f}%) ★未飽和裾野より外挿"
            )

        lines.append(
            f"  * 目標最大強度 (80%)  : {self.target_max:,.0f} counts "
            f"(デコンボリューション精度を最大化する安全最適値)"
        )

        if self.current_seconds is not None:
            acc_str = f" ({self.num_accumulations}回積算)" if self.num_accumulations > 1 else ""
            lines.append(f"  * 現行測定時間 (ASTS) : {self.current_seconds:.1f} 秒{acc_str}")
            if self.recommended_seconds is not None:
                lines.append("  * 【推奨測定時間】    :")
                pct = self.recommended_ratio * 100.0
                lines.append(
                    f"      >>> 目標露光時間   : {self.recommended_seconds:.1f} 秒 "
                    f"(現行の {pct:.0f}% に{advice_badge})"
                )
        else:
            lines.append("  * 【推奨露光倍率】    :")
            pct = self.recommended_ratio * 100.0
            lines.append(
                f"      >>> 設定推奨比率   : 現行測定時間の {pct:.0f}% "
                f"({advice_badge})"
            )

        if self.status == SaturationStatus.SATURATED:
            lines.append("  * 改善効果            : ピーク頭打ちによるリンギングを防止し、デブラー先鋭化を極限まで発揮")
        elif self.status == SaturationStatus.OPTIMAL:
            lines.append("  * 判定                : 現在の照射条件で最高のデブラー復元精度が得られています")
        elif self.status == SaturationStatus.UNDER_EXPOSED:
            lines.append("  * 改善効果            : 照射時間を延長することで微弱ピークの検出感度と結晶軸同定精度が向上")

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

    # 1. 観測最大値と位置の特定
    r_max_idx, th_max_idx = np.unravel_index(np.argmax(p_data), p_data.shape)
    obs_max = float(p_data[r_max_idx, th_max_idx])
    bg_val = float(np.median(p_data))
    peak_r = float(radii_mm[r_max_idx])
    peak_th = float(thetas_deg[th_max_idx])

    target_max = float(saturation_limit * target_ratio)

    # 2. 飽和判定
    # (a) 観測最大値が上限の 95% 以上
    # (b) または、ピーク近傍の半径・周方向で 90% 以上の値がプラトー (差分が極小) を形成
    is_saturated = obs_max >= (saturation_limit * 0.95)
    if not is_saturated and obs_max >= (saturation_limit * 0.90):
        # プラトーチェック
        r_slice = p_data[max(0, r_max_idx - 1) : min(p_data.shape[0], r_max_idx + 2), th_max_idx]
        if len(r_slice) >= 2 and np.max(np.abs(np.diff(r_slice))) < (obs_max * 0.015):
            is_saturated = True

    # 3. 半径 (r) 方向の未飽和裾野プロファイルからの真のピーク最大値外挿
    estimated_true_max = obs_max
    if is_saturated:
        try:
            w = 8
            r_indices = np.arange(max(0, r_max_idx - w), min(p_data.shape[0], r_max_idx + w + 1))
            prof_r = p_data[r_indices, th_max_idx]
            dr = (r_indices - r_max_idx).astype(np.float64)

            sig = prof_r - bg_val
            peak_sig = obs_max - bg_val

            # 未飽和スロープ (裾野) の抽出:
            # - 下限: 背景ノイズより十分高い (ピーク信号の 15% 以上)
            # - 上限: 飽和クリップ領域を除外 (上限の 88% 未満)
            valid = (sig > 0.15 * peak_sig) & (prof_r < saturation_limit * 0.88)

            if np.sum(valid) >= 3:
                # 対数パラボラ回帰: ln(sig) = a * dr^2 + b * dr + c
                y = np.log(np.maximum(sig[valid], 1.0))
                x = dr[valid]
                poly = np.polyfit(x, y, 2)
                a, b, c = poly
                if a < -1e-5:  # 上に凸の放物線
                    dr_peak = -b / (2.0 * a)
                    # ピーク位置が極端に離れていない場合のみ採用
                    if abs(dr_peak) < 2.5:
                        exponent = c - (b**2) / (4.0 * a)
                        # 数値オーバーフローガード (exp(50) ≒ 5.18e21)
                        if np.isfinite(exponent) and exponent < 50.0:
                            pred_sig = float(np.exp(exponent))
                            pred_max = bg_val + pred_sig
                            # 外挿値は少なくとも観測最大値以上
                            estimated_true_max = max(obs_max, pred_max)
        except Exception as e_fit:
            logger.debug("未飽和裾野フィッティング中の例外: %s", e_fit)
            estimated_true_max = obs_max

    # 4. ステータス決定
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

    # 6. 現行測定時間 (.ASTS) の取得と推奨秒数の計算
    curr_seconds: Optional[float] = None
    rec_seconds: Optional[float] = None
    num_acc = 1

    if csv_file is not None:
        meta = get_exposure_time_from_asts(csv_file)
        if meta is not None:
            curr_seconds = meta.total_seconds
            num_acc = meta.num_accumulations
            rec_seconds = max(1.0, round(curr_seconds * rec_ratio, 1))

    return ExposureOptimizationResult(
        status=status,
        observed_max=obs_max,
        estimated_true_max=estimated_true_max,
        background=bg_val,
        saturation_limit=saturation_limit,
        target_max=target_max,
        current_seconds=curr_seconds,
        recommended_seconds=rec_seconds,
        recommended_ratio=rec_ratio,
        peak_r_mm=peak_r,
        peak_theta_deg=peak_th,
        num_accumulations=num_acc,
    )
