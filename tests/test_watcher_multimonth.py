"""月替わり (年月サブフォルダ) 監視および resolve_watch_dir の単体テスト."""

import time
from pathlib import Path
import pytest

from astrolaue.watcher import resolve_watch_dir, ResidentWatcher


def test_resolve_watch_dir_normal(tmp_path):
    """通常の親ディレクトリ指定時はそのまま返されることを検証."""
    data_dir = tmp_path / "Data"
    data_dir.mkdir()
    assert resolve_watch_dir(data_dir) == data_dir
    assert resolve_watch_dir(None) is None


def test_resolve_watch_dir_promotes_month_subdir(tmp_path):
    """年月サブフォルダ (2610, 2611, 202611 等) が指定された際に親フォルダへ自動昇格することを検証."""
    data_dir = tmp_path / "LaueCamera" / "Data"
    data_dir.mkdir(parents=True)
    m2610 = data_dir / "2610"
    m2610.mkdir()
    m2611 = data_dir / "2611"
    m2611.mkdir()

    # 1. 2610 フォルダが指定された場合 -> 親の Data フォルダに昇格
    promoted = resolve_watch_dir(m2610)
    assert promoted == data_dir

    # 2. 2611 フォルダが指定された場合 -> 親の Data フォルダに昇格
    promoted_11 = resolve_watch_dir(m2611)
    assert promoted_11 == data_dir

    # 3. 存在しない 2612 フォルダが指定された場合 (未作成だが親 Data は存在) -> 親の Data フォルダ
    not_yet = data_dir / "2612"
    assert resolve_watch_dir(not_yet) == data_dir


def test_resolve_watch_dir_keeps_custom_subdir(tmp_path):
    """年月形式以外の個別ロットフォルダ等は親へ昇格せずそのまま維持されることを検証."""
    data_dir = tmp_path / "Data"
    custom_dir = data_dir / "Sample_Lot_A"
    custom_dir.mkdir(parents=True)
    assert resolve_watch_dir(custom_dir) == custom_dir


def test_resident_watcher_multimonth_detection(tmp_path):
    """月が変わって別サブフォルダ (2611) に CSV が保存されても自動検知・最新取得できることを検証."""
    base_data = tmp_path / "Data"
    oct_dir = base_data / "2610"
    nov_dir = base_data / "2611"
    oct_dir.mkdir(parents=True)
    nov_dir.mkdir(parents=True)

    # 10月の古い CSV
    old_csv = oct_dir / "261028_120000.CSV"
    old_csv.write_text("Dummy,CSV,Old", encoding="utf-8")

    # 11月の新着 CSV (mtime を後退させないよう少し待機)
    time.sleep(0.05)
    new_csv = nov_dir / "261105_150000.CSV"
    new_csv.write_text("Dummy,CSV,New", encoding="utf-8")

    # ユーザーが誤って 10月フォルダ (2610) を watch_dir に渡しても親の Data を監視
    watcher = ResidentWatcher(
        watch_dir=oct_dir,
        use_gui=False,
    )
    assert watcher.watch_dir == base_data

    # 監視対象内の全 CSV を再帰探索
    csvs = sorted(watcher.watch_dir.rglob("*.[cC][sS][vV]"), key=lambda f: f.stat().st_mtime)
    assert len(csvs) == 2
    # 最新 CSV が 11月フォルダのファイルであること
    assert csvs[-1] == new_csv
    assert csvs[-1].parent.name == "2611"
