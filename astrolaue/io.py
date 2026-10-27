"""モジュール: リガク (Rigaku) Astray / sLaue 生データ (CSV) 入出力.

測定装置が出力する極座標カウントマトリクス (.CSV) を高速・安全にパースし、
動径 r [mm], 方位角 theta [deg], 2theta 散乱角, カメラ距離 L [mm],
および正規化極座標配列を提供します。
"""

from __future__ import annotations

import csv
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SlaueRawData:
    """Rigaku Astray / sLaue の生測定データ構造体."""

    polar_data: np.ndarray
    """極座標生カウント配列。形状 (n_r, n_theta)、float32."""

    radii_mm: np.ndarray
    """動径座標配列 [mm]。形状 (n_r,)."""

    thetas_deg: np.ndarray
    """方位角座標配列 [deg]。形状 (n_theta,)."""

    two_theta_deg: Optional[np.ndarray] = None
    """2theta 散乱角配列 [deg]。形状 (n_r,)."""

    camera_length_mm: float = 27.0
    """幾何学的に特定されたカメラ長 L [mm]."""

    file_path: Optional[Path] = None
    """読み込み元ファイルパス."""

    def to_normalized_polar(
        self,
        bg_percentile: float = 5.0,
        high_percentile: float = 99.8,
    ) -> np.ndarray:
        """背景減算およびロバストパーセンタイル正規化を施した極座標配列 (0.0〜1.0) を生成します."""
        bg = float(np.percentile(self.polar_data, bg_percentile))
        sub = np.maximum(self.polar_data - bg, 0.0)
        p_hi = float(np.percentile(sub, high_percentile))
        denom = max(p_hi, 1e-4)
        norm = np.clip(sub / denom, 0.0, 1.0)
        return norm.astype(np.float32)


def load_slaue_csv(csv_path: Union[str, Path]) -> SlaueRawData:
    """Rigaku Astray / sLaue の極座標 CSV ファイルを読み込みます.

    CSV構造:
        - 1行目: ['', 'プロファイル']
        - 3行目: ['', '', '5.00 mm', '5.10 mm', ..., '40.40 mm'] (半径 r)
        - 4行目: ['', '', '169.51', '169.30', ..., '123.76'] (2theta 散乱角)
        - 5行目以降: ['', '0.00', '1218693', ...] (theta, カウント値...)

    Args:
        csv_path: CSVファイルパス

    Returns:
        SlaueRawData オブジェクト
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSVファイルが見つかりません: {path}")

    # Shift-JIS (CP932) で読み込み
    with open(path, "r", encoding="shift_jis", errors="replace") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 5:
        raise ValueError(f"CSVの行数が不足しています: {path} (行数={len(rows)})")

    # 3行目: 半径 r [mm]
    r_row = rows[2]
    r_headers = [c.replace(" mm", "").strip() for c in r_row if c.strip()]
    if not r_headers:
        raise ValueError(f"半径ヘッダ (3行目) を検出できませんでした: {path}")
    radii_mm = np.array([float(c) for c in r_headers], dtype=np.float32)

    # 4行目: 2theta 散乱角 [deg]
    two_theta_row = rows[3]
    two_theta_headers = [c.strip() for c in two_theta_row if c.strip()]
    two_thetas = None
    cam_len = 27.0
    if len(two_theta_headers) == len(r_headers):
        two_thetas = np.array([float(c) for c in two_theta_headers], dtype=np.float32)
        # 背面反射ラウエ幾何からカメラ距離 L を検証
        # tan(180 - 2theta) = r / L => L = r / tan(180 - 2theta)
        diff_angles = 180.0 - two_thetas
        valid_mask = diff_angles > 1.0
        if np.any(valid_mask):
            calc_L = radii_mm[valid_mask] / np.tan(np.radians(diff_angles[valid_mask]))
            cam_len = float(np.median(calc_L))

    # 5行目以降: theta [deg] と各 r におけるカウント値
    thetas = []
    matrix = []
    num_r = len(radii_mm)

    for r_idx, r in enumerate(rows[4:], start=5):
        if len(r) > 2 and r[1].strip():
            try:
                th = float(r[1])
                row_vals = [float(v) if v.strip() else 0.0 for v in r[2 : 2 + num_r]]
                if len(row_vals) < num_r:
                    row_vals.extend([0.0] * (num_r - len(row_vals)))
                thetas.append(th)
                matrix.append(row_vals)
            except ValueError:
                pass

    if not matrix:
        raise ValueError(f"有効な極座標データ行が存在しません: {path}")

    thetas_deg = np.array(thetas, dtype=np.float32)
    # (n_theta, n_r) から (n_r, n_theta) へ転置して AstroLaue の極座標規格に合わせる
    polar_data = np.array(matrix, dtype=np.float32).T

    logger.info(
        "sLaue生データを読み込みました: %s (r=%d点 [%.2f〜%.2f mm], theta=%d点 [%.2f〜%.2f°], カメラ長=%.2f mm)",
        path.name,
        polar_data.shape[0],
        radii_mm.min(),
        radii_mm.max(),
        polar_data.shape[1],
        thetas_deg.min(),
        thetas_deg.max(),
        cam_len,
    )

    return SlaueRawData(
        polar_data=polar_data,
        radii_mm=radii_mm,
        thetas_deg=thetas_deg,
        two_theta_deg=two_thetas,
        camera_length_mm=cam_len,
        file_path=path,
    )


@dataclass
class ExposureMetadata:
    """Rigaku sLaue 測定ログ (.ASTS) から取得された照射時間情報."""

    total_seconds: float
    """総照射時間 [秒] (積算がある場合は合計秒数)."""

    single_exposure_seconds: float
    """1回あたりの照射時間 [秒]."""

    num_accumulations: int
    """積算・照射回数."""

    asts_path: Optional[Path] = None
    """読み込み元 ASTS ファイルパス."""


def get_exposure_time_from_asts(
    path_or_csv: Union[str, Path]
) -> Optional[ExposureMetadata]:
    """同一フォルダの .ASTS ZIPアーカイブから X線照射時間 [秒] を高速に抽出します.

    Args:
        path_or_csv: .ASTS パス、または対応する .CSV パス

    Returns:
        ExposureMetadata オブジェクト。取得できなかった場合は None。
    """
    target = Path(path_or_csv)
    asts_path: Optional[Path] = None

    if target.suffix.lower() == ".asts" and target.is_file():
        asts_path = target
    elif target.suffix.lower() == ".csv":
        candidate = target.with_suffix(".ASTS")
        if candidate.is_file():
            asts_path = candidate
        else:
            candidate_lower = target.with_suffix(".asts")
            if candidate_lower.is_file():
                asts_path = candidate_lower

    if asts_path is None or not asts_path.is_file():
        return None

    if not zipfile.is_zipfile(asts_path):
        return None

    try:
        with zipfile.ZipFile(asts_path, "r") as z:
            names = z.namelist()
            # 1. ResultLog.txt から 'X線照射 (○○秒)' を検索 (Shift-JIS / CP932)
            if "ResultLog.txt" in names:
                raw_bytes = z.read("ResultLog.txt")
                for enc in ["cp932", "shift_jis", "utf-8"]:
                    try:
                        text = raw_bytes.decode(enc)
                        matches = re.findall(r"X線照射 \(([0-9]+(?:\.[0-9]+)?)秒\)", text)
                        if matches:
                            durations = [float(m) for m in matches]
                            tot_sec = sum(durations)
                            single_sec = durations[0] if durations else tot_sec
                            return ExposureMetadata(
                                total_seconds=tot_sec,
                                single_exposure_seconds=single_sec,
                                num_accumulations=len(durations),
                                asts_path=asts_path,
                            )
                    except UnicodeDecodeError:
                        continue

            # 2. MeasureCtrLog.txt からタイムスタンプ差分による照射時間を検索
            if "MeasureCtrLog.txt" in names:
                raw_bytes = z.read("MeasureCtrLog.txt")
                for enc in ["cp932", "shift_jis", "utf-8"]:
                    try:
                        text = raw_bytes.decode(enc)
                        start_matches = re.findall(
                            r"(\d{2}:\d{2}:\d{2}) : CMeasExecIrradiationXray::Measure , Start",
                            text,
                        )
                        end_matches = re.findall(
                            r"(\d{2}:\d{2}:\d{2}) : CMeasExecIrradiationXray::Measure , End",
                            text,
                        )
                        if start_matches and end_matches and len(start_matches) == len(end_matches):
                            durations = []
                            for st_str, ed_str in zip(start_matches, end_matches):
                                try:
                                    h1, m1, s1 = map(int, st_str.split(":"))
                                    h2, m2, s2 = map(int, ed_str.split(":"))
                                    diff = (h2 * 3600 + m2 * 60 + s2) - (h1 * 3600 + m1 * 60 + s1)
                                    if diff < 0:
                                        diff += 86400  # 日跨ぎ対応 (24時間加算)
                                    if diff > 0:
                                        durations.append(float(diff))
                                except ValueError:
                                    pass
                            if durations:
                                tot_sec = sum(durations)
                                return ExposureMetadata(
                                    total_seconds=tot_sec,
                                    single_exposure_seconds=durations[0],
                                    num_accumulations=len(durations),
                                    asts_path=asts_path,
                                )
                    except UnicodeDecodeError:
                        continue
    except Exception as e_zip:
        logger.debug("ASTS 露光時間パース中の例外: %s", e_zip)

    return None
