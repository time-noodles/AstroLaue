#!/usr/bin/env python3
"""AstroLaue 簡単起動対話型ランチャー (Windows & Linux クロスプラットフォーム対応).

メニュー選択またはドラッグ＆ドロップにより、CLIオプションを意識せずに
小型ラウエ画像復元パイプラインを実行できます。
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

# 自モジュール検索用
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


def find_python() -> str:
    """利用可能な Python 実行可能ファイルのパスを自動探索します."""
    # ユーザー指定の miniconda py312 や仮想環境を最優先探索
    candidates = [
        # Windows miniconda py312
        Path.home() / "miniconda3" / "envs" / "py312" / "python.exe",
        Path.home() / "Miniconda3" / "envs" / "py312" / "python.exe",
        Path.home() / "AppData" / "Local" / "miniconda3" / "envs" / "py312" / "python.exe",
        Path("C:/miniconda3/envs/py312/python.exe"),
        Path("C:/ProgramData/miniconda3/envs/py312/python.exe"),
        Path.home() / "anaconda3" / "envs" / "py312" / "python.exe",
        Path.home() / "Anaconda3" / "envs" / "py312" / "python.exe",
        # Linux miniconda py312
        Path.home() / "miniconda3" / "envs" / "py312" / "bin" / "python",
        # ローカル .venv
        SCRIPT_DIR / ".venv" / "bin" / "python",
        SCRIPT_DIR / ".venv" / "Scripts" / "python.exe",
        Path.home() / ".venvs" / "astrolaue" / "bin" / "python",
        Path.home() / ".venvs" / "astrolaue" / "Scripts" / "python.exe",
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)

    # 2. 現在実行中の Python
    return sys.executable



def open_folder(path: Path) -> None:
    """指定フォルダをOS標準のファイルマネージャー (エクスプローラー等) で開きます."""
    try:
        p_str = str(path.resolve())
        if sys.platform == "win32":
            os.startfile(p_str)
        elif sys.platform == "darwin":
            subprocess.run(["open", p_str], check=False)
        else:
            subprocess.run(["xdg-open", p_str], check=False)
    except Exception:
        pass


def print_banner() -> None:
    banner = """
=============================================================
        🌟 AstroLaue - 小型ラウエ画像復元パイプライン 🌟
         (Azimuthal Motion Blur Sharpening Pipeline)
=============================================================
"""
    print(banner)


def menu_single_image(py_exe: str) -> None:
    print("\n--- [1] 単一画像ファイルの復元・解析 ---")
    default_img = SCRIPT_DIR / "share" / "画像260925.png"
    prompt = f"画像ファイルパスを入力 (未入力の場合はデフォルト '{default_img.name}'): "
    user_input = input(prompt).strip()

    if not user_input:
        target = default_img
    else:
        # クォート等の除去
        clean_path = user_input.strip("\"' ")
        target = Path(clean_path)

    if not target.is_file():
        print(f"❌ エラー: ファイルが見つかりません: {target}")
        return

    out_dir = SCRIPT_DIR / "results"
    cmd = [
        py_exe,
        str(SCRIPT_DIR / "main.py"),
        "--input", str(target),
        "--output", str(out_dir),
        "--show-diagnostic",
        "--export-transparent",
        "--validate-physics",
    ]

    print(f"\n▶ 実行中: {target.name} ...")
    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print(f"\n✅ 完了! 出力先: {out_dir}")
        open_folder(out_dir)
    else:
        print("\n❌ 処理中にエラーが発生しました。")


def menu_batch_process(py_exe: str) -> None:
    print("\n--- [2] ディレクトリ内全画像の一括バッチ処理 ---")
    default_dir = SCRIPT_DIR / "share"
    prompt = f"対象画像ディレクトリを入力 (未入力の場合は '{default_dir.name}'): "
    user_input = input(prompt).strip()

    if not user_input:
        target_dir = default_dir
    else:
        clean_path = user_input.strip("\"' ")
        target_dir = Path(clean_path)

    if not target_dir.is_dir():
        print(f"❌ エラー: ディレクトリが見つかりません: {target_dir}")
        return

    out_dir = SCRIPT_DIR / "results_batch"
    cmd = [
        py_exe,
        str(SCRIPT_DIR / "main.py"),
        "--batch-dir", str(target_dir),
        "--output", str(out_dir),
        "--export-transparent",
        "--validate-physics",
    ]

    print(f"\n▶ 一括処理を実行中: {target_dir} ...")
    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print(f"\n✅ バッチ処理完了! 出力先: {out_dir}")
        print(f"・サマリーCSV: {out_dir / 'summary_report.csv'}")
        print(f"・一覧比較図 : {out_dir / 'batch_summary_plot.png'}")
        open_folder(out_dir)
    else:
        print("\n❌ バッチ処理中にエラーが発生しました。")


def menu_live_capture(py_exe: str) -> None:
    print("\n--- [3] 画面のライブキャプチャ & 復元 (1回実行) ---")
    print("手前に表示されているウィンドウまたは画面全体から回折パターンを自動検出します。")
    title = input("特定のウィンドウタイトルを指定する場合のみ入力 (未入力で画面全体/最前面): ").strip()

    out_dir = SCRIPT_DIR / "results_live"
    cmd = [
        py_exe,
        str(SCRIPT_DIR / "main.py"),
        "--live",
        "--output", str(out_dir),
        "--show-diagnostic",
        "--export-transparent",
    ]
    if title:
        cmd.extend(["--window-title", title])

    print("\n▶ 画面キャプチャ＆復元を実行中...")
    ret = subprocess.run(cmd)
    if ret.returncode == 0:
        print(f"\n✅ 完了! 出力先: {out_dir}")
        open_folder(out_dir)
    else:
        print("\n❌ ライブキャプチャ中にエラーが発生しました。")


def menu_resident_watcher(py_exe: str) -> None:
    print("\n--- [4] 常駐ホットキー監視モード (Resident Watcher) ---")
    print("バックグラウンドでホットキーを常時待機し、キー押下で即座にキャプチャ・復元します。")
    print("測定ソフトや画像ビューアを画面に表示した状態で [F9] を押してください。")
    print("-------------------------------------------------------------")
    out_dir = SCRIPT_DIR / "results_live"
    cmd = [
        py_exe,
        str(SCRIPT_DIR / "main.py"),
        "--watch",
        "--output", str(out_dir),
        "--show-diagnostic",
        "--export-transparent",
    ]
    subprocess.run(cmd)


def menu_run_tests(py_exe: str) -> None:
    print("\n--- [5] テストスイートの実行 (pytest) ---")
    pytest_candidates = [
        Path(py_exe).parent / "pytest",
        Path(py_exe).parent / "pytest.exe",
    ]
    pytest_bin = None
    for cand in pytest_candidates:
        if cand.is_file():
            pytest_bin = str(cand)
            break

    if pytest_bin:
        cmd = [pytest_bin, "-v"]
    else:
        cmd = [py_exe, "-m", "pytest", "-v"]

    print("▶ テストを実行中...")
    subprocess.run(cmd, cwd=str(SCRIPT_DIR))


def main() -> None:
    py_exe = find_python()

    # コマンドライン引数に画像ファイルが渡された場合 (ドラッグ＆ドロップ対応)
    if len(sys.argv) > 1:
        arg_path = Path(sys.argv[1].strip("\"' "))
        if arg_path.is_file():
            print_banner()
            print(f"🎯 ドラッグ＆ドロップされた画像を検出しました: {arg_path.name}")
            out_dir = SCRIPT_DIR / "results"
            cmd = [
                py_exe,
                str(SCRIPT_DIR / "main.py"),
                "--input", str(arg_path),
                "--output", str(out_dir),
                "--show-diagnostic",
                "--export-transparent",
                "--validate-physics",
            ]
            subprocess.run(cmd)
            open_folder(out_dir)
            input("\nEnterキーを押して終了してください...")
            return
        elif arg_path.is_dir():
            print_banner()
            print(f"🎯 ドラッグ＆ドロップされたディレクトリを検出しました: {arg_path.name}")
            out_dir = SCRIPT_DIR / "results_batch"
            cmd = [
                py_exe,
                str(SCRIPT_DIR / "main.py"),
                "--batch-dir", str(arg_path),
                "--output", str(out_dir),
                "--export-transparent",
                "--validate-physics",
            ]
            subprocess.run(cmd)
            open_folder(out_dir)
            input("\nEnterキーを押して終了してください...")
            return

    while True:
        print_banner()
        print("実行したい操作を選択してください:")
        print("  [1] 単一画像ファイルの復元・解析")
        print("  [2] /share 内全画像の一括バッチ処理")
        print("  [3] 画面のライブキャプチャ＆復元 (1回実行)")
        print("  [4] 常駐ホットキー監視モード (F9でいつでも自動キャプチャ＆復元)")
        print("  [5] テストスイートの実行 (pytest)")
        print("  [0] 終了")
        print("-------------------------------------------------------------")

        choice = input("選択 (0-5): ").strip()

        if choice == "1":
            menu_single_image(py_exe)
        elif choice == "2":
            menu_batch_process(py_exe)
        elif choice == "3":
            menu_live_capture(py_exe)
        elif choice == "4":
            menu_resident_watcher(py_exe)
        elif choice == "5":
            menu_run_tests(py_exe)
        elif choice in ("0", "q", "quit", "exit"):
            print("\n終了します。お疲れ様でした！")
            break
        else:
            print("無効な選択です。0〜5の数字を入力してください。")

        input("\nEnterキーを押してメニューに戻ります...")


if __name__ == "__main__":
    main()
