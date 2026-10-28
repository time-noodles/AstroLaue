"""常駐ホットキー監視モジュール (Resident Hotkey Watcher).

測定ごとに毎回バッチファイルを起動する手間を省くため、
バックグラウンドでホットキーを常時監視し、キー押下をトリガーとして
即座に画面キャプチャ＆ラウエ画像復元パイプラインを実行します。

特徴:
- Windows標準ショートカットと重複しない [F9] および [Ctrl + Shift + L] を監視。
- 特定のアプリケーションに依存せず、現在手前のアクティブウィンドウまたは画面全体から
  回折パターンを自動検出・切り出し。
- 復元完了時にビープ音で通知し、最新結果を results_live/ に保存すると同時に
  results_live/history/ にタイムスタンプ付きで蓄積保存。
- Windows標準 ctypes API のみで動作し、追加パッケージのインストールは不要。
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Literal
import shutil

from astrolaue.capture import capture_window
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig
from astrolaue.alignment import (
    LaueAlignmentConfig,
    solve_laue_alignment,
    plot_alignment_diagnostics,
)
from astrolaue.clipboard import copy_image_to_clipboard, copy_text_to_clipboard

logger = logging.getLogger(__name__)


def disable_windows_quick_edit() -> None:
    """Windows コンソールの QuickEdit モードを無効化し、マウスクリックによるフリーズを防止します."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h_stdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE = -10
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h_stdin, ctypes.byref(mode)):
            ENABLE_QUICK_EDIT_MODE = 0x0040
            ENABLE_EXTENDED_FLAGS = 0x0080
            new_mode = (mode.value & ~ENABLE_QUICK_EDIT_MODE) | ENABLE_EXTENDED_FLAGS
            kernel32.SetConsoleMode(h_stdin, new_mode)
            logger.info("Windows コンソールの QuickEdit モードを無効化しました")
    except Exception as e:
        logger.debug("QuickEdit 無効化例外: %s", e)


def find_default_watch_dir() -> Optional[Path]:
    """測定PC上の LaueCamera 生データ保存フォルダ (例: Data/2610) を自動探索します."""
    # 1. path.txt のチェック (作業ディレクトリ or share/slaue-data/)
    for cand in [Path("path.txt"), Path("share/slaue-data/path.txt")]:
        if cand.is_file():
            try:
                line = cand.read_text(encoding="utf-8", errors="replace").strip()
                if line:
                    p = Path(line)
                    if p.is_dir():
                        return p
                    # Windowsパスが書かれている場合
                    if "\\" in line:
                        p_win = Path(line.replace("\\", "/"))
                        if p_win.is_dir():
                            return p_win
            except Exception:
                pass

    # 2. 標準ドキュメントフォルダ探索 (Windows / Linux)
    candidates_base = []
    if sys.platform == "win32":
        user_profile = os.environ.get("USERPROFILE", "")
        if user_profile:
            candidates_base.append(Path(user_profile) / "Documents" / "LaueCamera" / "Data")
            candidates_base.append(Path(user_profile) / "Documents" / "u-X360" / "Data")
    home = Path.home()
    candidates_base.append(home / "Documents" / "LaueCamera" / "Data")
    candidates_base.append(home / "Documents" / "u-X360" / "Data")
    candidates_base.append(Path("share/slaue-data"))

    for base in candidates_base:
        if base.is_dir():
            # 年月フォルダ (例: 2610, 2609...) のうち最新のものを探索
            subdirs = [d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]
            if subdirs:
                subdirs.sort(key=lambda d: d.name)
                return subdirs[-1]
            return base

    return None


class ResidentWatcher:
    """ホットキー監視＆生データフォルダ監視による常駐自動復元・結晶方位アライメントマネージャー."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        output_dir: str | Path = "./results_live",
        watch_dir: Optional[str | Path] = None,
        window_title: Optional[str] = None,
        monitor_index: int = 1,
        target_mode: str = "auto",
        hotkey_name: str = "F9",
        camera_length_mm: float = 27.0,
        pixel_size_mm: float = 0.05,
        stage_order: Literal["XY", "YX"] = "XY",
        sign_x: int = 1,
        sign_y: int = -1,
        alignment_config: Optional[LaueAlignmentConfig] = None,
        use_gui: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config or PipelineConfig(output_dir=self.output_dir)
        self.config.output_dir = self.output_dir
        self.pipeline = AstroLauePipeline(self.config)
        self.history_dir = self.output_dir / "history"
        self.window_title = window_title
        self.monitor_index = monitor_index
        self.target_mode = target_mode
        self.hotkey_name = hotkey_name
        self.use_gui = use_gui
        self.gui: Optional[Any] = None
        self._running = False
        self._is_processing = False
        self._lock = threading.Lock()

        # 監視対象フォルダの初期化
        if watch_dir:
            self.watch_dir = Path(watch_dir)
        else:
            self.watch_dir = find_default_watch_dir()

        self._seen_csv_files: dict[str, float] = {}

        # 結晶方位アライメント設定
        self.alignment_config = alignment_config or LaueAlignmentConfig(
            camera_length_mm=camera_length_mm,
            pixel_size_mm=pixel_size_mm,
            stage_order=stage_order,
            sign_x=sign_x,
            sign_y=sign_y,
        )

        # 出力ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def trigger_capture_and_restore(self) -> bool:
        """キャプチャを実行し、AstroLaue復元パイプラインを走らせます."""
        if not self._lock.acquire(blocking=False):
            print("\n[!] 前回の復元処理が実行中です。完了をお待ちください...")
            return False

        try:
            print("\n" + "=" * 60)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[*] キャプチャ＆復元を開始します ({now_str})")
            print("=" * 60)

            # 出力ディレクトリ直下の古いファイルを初期化クリーンアップ (history/ は保護)
            if self.output_dir.exists():
                for f in self.output_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".png", ".csv", ".jpg"]:
                        try:
                            f.unlink(missing_ok=True)
                        except Exception:
                            pass

            # 1. 画面またはアクティブウィンドウからキャプチャ
            print("[1/3] 画面を取得中...")
            frame = capture_window(
                window_title_pattern=self.window_title,
                monitor_index=self.monitor_index,
                target_mode=self.target_mode,
            )

            # 2. AstroLaue復元パイプライン実行 (内部でROI自動検出・幾何補正・逆畳み込み・解析・履歴保存・クリップボードコピーを完結)
            print("[2/3] 回折パターン検出 & ブラー逆畳み込み実行中...")
            result = self.pipeline.run(image_source=frame)

            # 4. 結果サマリー表示
            print("[3/3] 復元完了!")
            print("-" * 60)
            print(f"  * 推定ブラー長 Lθ : {result.blur_length:.2f} px")
            if result.spots:
                reductions = [s.fwhm_reduction_ratio for s in result.spots if s.fwhm_before > 0]
                avg_red = sum(reductions) / len(reductions) if reductions else 0.0
                print(f"  * 平均FWHM先鋭化率: {avg_red:.1f} %")
                print(f"  * 検出スポット数  : {len(result.spots)} 個")
            if result.physics_report:
                flux = result.physics_report.mean_flux_ratio
                shift = result.physics_report.mean_centroid_shift_px
                print(f"  * 光量保存比 (Flux): {flux:.3f} (理論値: 1.000)")
                print(f"  * 重心シフト      : {shift:.2f} px")
            print(f"  * 保存先フォルダ  : {self.output_dir.resolve()}")
            print("  * 直下の最新成果物一覧:")
            if self.output_dir.exists():
                direct_files = [f for f in self.output_dir.iterdir() if f.is_file()]
                for df in sorted(direct_files):
                    mtime_str = datetime.datetime.fromtimestamp(df.stat().st_mtime).strftime("%H:%M:%S")
                    size_kb = df.stat().st_size / 1024.0
                    print(f"    - {df.name:<34} (更新: {mtime_str}, {size_kb:6.1f} KB)")
            if result.history_png_path:
                print(f"  * 履歴永続保存    : {result.history_png_path.resolve()}")
            print("-" * 60)

            # 成功通知音 (Windows)
            self._notify_sound(success=True)

            print("\n>>> 次の操作を待機中... ([Enter]: 最新CSV再解析, [S]: 画面キャプチャ復元, [A]: 角度アライメント, [Q]: 終了) <<<", flush=True)
            return True

        except Exception as ex:
            import traceback
            tb_str = traceback.format_exc()
            logger.error("キャプチャ＆復元中にエラーが発生しました: %s\n%s", ex, tb_str)
            print(f"\n[ERROR] 復元処理に失敗しました: {ex}", flush=True)
            tb_lines = [l for l in tb_str.strip().splitlines() if l.strip()]
            if len(tb_lines) > 1:
                print(f"  詳細: {tb_lines[-2]} -> {tb_lines[-1]}", flush=True)
            self._notify_sound(success=False)
            print("\n>>> 待機中... ([Enter]: 最新CSV再解析, [S]: 画面キャプチャ復元, [A]: 角度アライメント) <<<", flush=True)
            return False

        finally:
            self._lock.release()

    def trigger_restore_from_csv(self, csv_path: Path) -> bool:
        """sLaue CSV生データから直接極座標復元＆結晶方位アライメントを自動実行します."""
        if not self._lock.acquire(blocking=False):
            print(f"\n[!] 処理中のため、新着CSV ({csv_path.name}) の処理を保留し次回ループで再試行します...", flush=True)
            # ロック競合時は追跡リストから除外して次回ループで再検知可能にする
            try:
                self._seen_csv_files.pop(str(csv_path.resolve()), None)
            except Exception:
                pass
            return False

        try:
            print("\n" + "=" * 60)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[*] 新着生データ検知: {csv_path.name} ({now_str})")
            print(f"    (保存元: {csv_path.parent})")
            print("=" * 60)

            # 出力ディレクトリ直下の古いファイルを初期化クリーンアップ (history/ は保護)
            if self.output_dir.exists():
                for f in self.output_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in [".png", ".csv", ".jpg"]:
                        try:
                            f.unlink(missing_ok=True)
                        except Exception:
                            pass

            print("[1/2] 極座標生カウント読み込み & 超高速アライメント＆露光解析中...")
            if self.gui is not None:
                self.gui.update_status(f"⚡ 解析中: {csv_path.name}", color="#ffd700", filename=csv_path.name)
            t_start = time.time()
            alignment_done = False

            def _on_fast_feedback(
                align_res: LaueAlignmentResult,
                exp_res: Optional[ExposureOptimizationResult] = None,
            ) -> None:
                nonlocal alignment_done
                elapsed = (time.time() - t_start) * 1000.0
                best = align_res.best_candidate
                align_text = f"Δθx = {best.delta_theta_x_deg:+.2f}°, Δθy = {best.delta_theta_y_deg:+.2f}°"

                # セクション 1: 【結晶主軸アライメント】(独立表示)
                print("\n" + "=" * 70)
                print(f"【⚡ 結晶主軸アライメント速報 ({elapsed:.0f} ms)】")
                print(f"  * 推定 (001) 主軸座標  : 検出器 (X = {best.detector_x_mm:+.2f} mm, Y = {best.detector_y_mm:+.2f} mm)")
                print(f"  * 中心軸からの全傾き角 : ψ = {best.total_tilt_deg:.2f}°")
                print("  * 【推奨ゴニオ補正角】 :")
                print(f"      >>> Δθx (X軸回転) = {best.delta_theta_x_deg:+.2f}°")
                print(f"      >>> Δθy (Y軸回転) = {best.delta_theta_y_deg:+.2f}°")
                print(f"  * クリップボード       : 補正角度 ({align_text}) を転送完了")
                print("=" * 70)
                copy_text_to_clipboard(align_text)

                # セクション 2: 【露光時間最適化＆ピーク解析】(独立表示)
                if exp_res is not None:
                    print("\n" + exp_res.format_section() + "\n")

                # 最前面 GUI HUD へのリアルタイム速報反映
                if self.gui is not None:
                    self.gui.update_alignment(align_res, source_name=csv_path.name)
                    if exp_res is not None:
                        self.gui.update_exposure(exp_res)

                self._notify_sound(success=True)
                alignment_done = True

            result = self.pipeline.run(image_source=csv_path, on_alignment_ready=_on_fast_feedback)

            print("[2/2] 高解像度画像出力＆履歴保存完了!")
            print("-" * 60)
            print(f"  * 入力データ      : {csv_path.name}")
            print(f"  * 推定ブラー長 Lθ : {result.blur_length:.2f} px")

            # 結晶主軸アライメント結果の表示 (速報で未表示の場合のフォールバック)
            if result.alignment_result and not alignment_done:
                best = result.alignment_result.best_candidate
                align_text = f"Δθx = {best.delta_theta_x_deg:+.2f}°, Δθy = {best.delta_theta_y_deg:+.2f}°"
                print("\n" + "=" * 70)
                print("【結晶主軸アライメント結果】")
                print(f"  * 【推奨ゴニオ補正角】: {align_text} (全傾き角: {best.total_tilt_deg:.2f}°)")
                copy_text_to_clipboard(align_text)
                print(f"    -> 補正角度 ({align_text}) をクリップボードにコピーしました！")
                print("=" * 70)

                if result.exposure_result:
                    print("\n" + result.exposure_result.format_section())

            if result.spots:
                reductions = [s.fwhm_reduction_ratio for s in result.spots if s.fwhm_before > 0]
                avg_red = sum(reductions) / len(reductions) if reductions else 0.0
                print(f"  * 平均FWHM先鋭化率: {avg_red:.1f} %")
                print(f"  * 検出スポット数  : {len(result.spots)} 個")

            print(f"  * 保存先フォルダ  : {self.output_dir.resolve()}")
            print("  * 直下の最新成果物一覧:")
            if self.output_dir.exists():
                direct_files = [f for f in self.output_dir.iterdir() if f.is_file()]
                for df in sorted(direct_files):
                    mtime_str = datetime.datetime.fromtimestamp(df.stat().st_mtime).strftime("%H:%M:%S")
                    size_kb = df.stat().st_size / 1024.0
                    print(f"    - {df.name:<34} (更新: {mtime_str}, {size_kb:6.1f} KB)")
            if result.history_png_path:
                print(f"  * 履歴永続保存    : {result.history_png_path.resolve()}")
            print("-" * 60)

            # 成功通知音
            self._notify_sound(success=True)
            if self.gui is not None:
                self.gui.update_status("✅ 復元・アライメント完了", color="#69f0ae", filename=csv_path.name)
            print("\n>>> 自動監視中... (新着CSV常時待機 / [Enter]: 最新CSV再解析, [S]: キャプチャ, [Q]: 終了) <<<", flush=True)
            return True

        except Exception as ex:
            import traceback
            tb_str = traceback.format_exc()
            logger.error("CSV復元中にエラーが発生しました: %s\n%s", ex, tb_str)
            print(f"\n[ERROR] CSV復元処理に失敗しました: {ex}", flush=True)
            tb_lines = [l for l in tb_str.strip().splitlines() if l.strip()]
            if len(tb_lines) > 1:
                print(f"  詳細: {tb_lines[-2]} -> {tb_lines[-1]}", flush=True)
            self._notify_sound(success=False)
            if self.gui is not None:
                self.gui.update_status(f"❌ 解析失敗: {ex}", color="#ff5252", filename=csv_path.name)
            print("\n>>> 自動監視中... (新着CSV常時待機 / [Enter]: 最新CSV再解析, [S]: キャプチャ) <<<", flush=True)
            return False

        finally:
            self._lock.release()

    def trigger_crystal_alignment(self) -> bool:
        """画面キャプチャから結晶主軸自動同定＆2軸ゴニオ補正角推定を高速実行します."""
        if not self._lock.acquire(blocking=False):
            print("\n[!] 前回の処理が実行中です。完了をお待ちください...")
            return False

        try:
            print("\n" + "=" * 60)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[*] 結晶方位アライメントを開始します ({now_str})")
            print(f"    (カメラ長 L={self.alignment_config.camera_length_mm:.1f} mm, ステージ={self.alignment_config.stage_order})")
            print("=" * 60)

            # 1. 画面またはアクティブウィンドウからキャプチャ
            print("[1/2] 画面を取得中...")
            frame = capture_window(
                window_title_pattern=self.window_title,
                monitor_index=self.monitor_index,
                target_mode=self.target_mode,
            )

            # 2. アライメント計算実行 (背景減算・ピーク抽出・グノモン直線探索・交点推定)
            print("[2/2] スポット抽出 & グノモン晶帯解析実行中...")
            t0 = time.perf_counter()
            res = solve_laue_alignment(frame, config=self.alignment_config)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # 3. 診断図の出力・保存
            diag_live_path = self.output_dir / "laue_alignment_live.png"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            diag_history_path = self.history_dir / f"laue_alignment_{timestamp}.png"

            plot_alignment_diagnostics(res, save_path=diag_live_path, show=False)
            try:
                shutil.copy(diag_live_path, diag_history_path)
            except Exception:
                pass

            # クリップボードへコピー (画像およびテキスト)
            copy_image_to_clipboard(diag_live_path)
            angle_text = f"Δθx={res.delta_theta_x_deg:+.2f}°, Δθy={res.delta_theta_y_deg:+.2f}° (Total tilt: {res.total_tilt_deg:.2f}°)"
            copy_text_to_clipboard(angle_text)

            # 4. コンソール結果サマリー表示
            print("-" * 60)
            print("【結晶方位アライメント結果】")
            print(f"  * 処理時間             : {elapsed_ms:.1f} ms")
            print(f"  * 検出スポット数       : {len(res.spots_xy_mm)} 個")
            print(f"  * 抽出された主要晶帯   : {len(res.zones)} 本")
            for z in res.zones[:4]:
                print(f"      - 晶帯 #{z.line_id}: インライア={z.num_inliers}点, スコア={z.score:.2f}")
            print(f"  * 推定 (001) 主軸座標  : 検出器 (X={res.best_candidate.detector_x_mm:+.2f}, Y={res.best_candidate.detector_y_mm:+.2f}) mm")
            print(f"  * 中心軸からの全傾き角 : ψ = {res.total_tilt_deg:.2f}°")
            print("  * 【推奨ゴニオ補正角】 :")
            print(f"      >>> Δθx (X軸回転) = {res.delta_theta_x_deg:+.2f}°")
            print(f"      >>> Δθy (Y軸回転) = {res.delta_theta_y_deg:+.2f}°")
            if len(res.candidates) > 1:
                print("  * 代替主軸候補:")
                for c in res.candidates[1:3]:
                    print(f"      - 第{c.rank}候補: Δθx={c.delta_theta_x_deg:+.2f}°, Δθy={c.delta_theta_y_deg:+.2f}°, スコア={c.score:.2f}")
            print(f"  * 診断図保存           : {diag_live_path.resolve()}")
            print(f"  * 履歴永続保存         : {diag_history_path.resolve()}")
            print(f"  * クリップボード       : 診断図 & 補正角テキストを転送完了")
            print("-" * 60)

            # 成功通知音
            self._notify_sound(success=True)
            if self.gui is not None:
                self.gui.update_alignment(res, source_name="Screen Capture")

            print("\n>>> 次の操作を待機中... ([Enter]: 最新CSV再解析, [S]: 画面キャプチャ復元, [A]: 角度アライメント, [Q]: 終了) <<<", flush=True)
            return True

        except Exception as ex:
            import traceback
            tb_str = traceback.format_exc()
            logger.error("アライメント実行中にエラーが発生しました: %s\n%s", ex, tb_str)
            print(f"\n[ERROR] アライメント処理に失敗しました: {ex}", flush=True)
            tb_lines = [l for l in tb_str.strip().splitlines() if l.strip()]
            if len(tb_lines) > 1:
                print(f"  詳細: {tb_lines[-2]} -> {tb_lines[-1]}", flush=True)
            self._notify_sound(success=False)
            print("\n>>> 待機中... ([Enter]: 最新CSV再解析, [S]: 画面キャプチャ復元, [A]: 角度アライメント) <<<", flush=True)
            return False

        finally:
            self._lock.release()

    def trigger_restore_latest_csv(self) -> bool:
        """監視フォルダ内 (サブフォルダ含む) の最も新しいCSVファイルを再読み込みして復元を実行します."""
        if not self.watch_dir or not self.watch_dir.is_dir():
            print("\n[!] 監視フォルダが設定されていないため、最新CSVを取得できません。", flush=True)
            return False

        csv_files = [f for f in self.watch_dir.rglob("*.[cC][sS][vV]") if f.is_file()]
        if not csv_files:
            print(f"\n[!] 監視フォルダ ({self.watch_dir}) 内に CSV ファイルが見つかりません。", flush=True)
            return False

        csv_files.sort(key=lambda p: p.stat().st_mtime)
        latest_csv = csv_files[-1]
        print(f"\n[*] 直近の測定CSV ({latest_csv.name}) を再解析します...", flush=True)
        return self.trigger_restore_from_csv(latest_csv)

    def _notify_sound(self, success: bool = True) -> None:
        """処理完了をビープ音で通知します."""
        if sys.platform == "win32":
            try:
                import winsound
                if success:
                    # 心地よい2音チャイム (1000Hz -> 1500Hz)
                    winsound.Beep(1046, 120)  # C6
                    time.sleep(0.05)
                    winsound.Beep(1318, 150)  # E6
                else:
                    # エラー低音 (400Hz)
                    winsound.Beep(440, 300)
                return
            except Exception:
                pass
        # フォールバック
        print("\a", end="", flush=True)

    def _print_help(self) -> None:
        """キー操作ヘルプを表示します."""
        print("\n" + "=" * 65, flush=True)
        print("  【AstroLaue ライブ監視コマンド一覧】", flush=True)
        print("    ★ 常時自動監視 : 測定ソフトで測定完了 (.CSV 保存) 時に全自動復元＆角度推定", flush=True)
        print("    [Enter] / [C]   : 直近の最新測定 CSV を再解析・再復元", flush=True)
        print("    [S] + [Enter]   : 画面キャプチャによる即時手動復元 (表示中ウィンドウ)", flush=True)
        print("    [A] + [Enter]   : 画面キャプチャからの結晶主軸アライメント (高速角度推定)", flush=True)
        print("    [F9]  (Windows) : グローバルホットキー: 画面キャプチャ復元", flush=True)
        print("    [F10] (Windows) : グローバルホットキー: 結晶主軸アライメント", flush=True)
        print("    [H] + [Enter]   : このヘルプを表示", flush=True)
        print("    [Q] + [Enter]   : ライブ監視を終了", flush=True)
        print("=" * 65 + "\n", flush=True)

    def stop(self) -> None:
        """常駐監視を終了します."""
        self._running = False
        if self.gui is not None and getattr(self.gui, "root", None) is not None:
            try:
                self.gui.root.after(0, self.gui.root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        """常駐監視ループを開始します."""
        self._running = True
        disable_windows_quick_edit()
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

        print("=" * 65, flush=True)
        print("  AstroLaue: リアルタイム・ライブ監視モード (Live Watcher)", flush=True)
        print("=" * 65, flush=True)
        if self.watch_dir and self.watch_dir.is_dir():
            print(f"  * 【生データ常時監視】: {self.watch_dir.resolve()}", flush=True)
            print("    -> サブフォルダ含む新着 .CSV 保存時に自動で完全復元＆角度推定！", flush=True)
        else:
            print("  * (生データ自動監視フォルダ未設定: 手動キャプチャまたは --watch-dir で指定可能)", flush=True)
        print("  * 【コマンド操作】", flush=True)
        print("    - 自動監視     : 新着CSVファイルを常時バックグラウンド待機中", flush=True)
        print("    - [Enter] / [C]: 直近の最新測定 CSV を再解析・再復元", flush=True)
        print("    - [S] + Enter  : 画面キャプチャ即時復元 (特定アプリに依存せず自動切り出し)", flush=True)
        print("    - [A] + Enter  : 画面キャプチャから結晶主軸アライメント", flush=True)
        print("    - [Q] + Enter  : 終了", flush=True)
        print("  * 復元結果・アライメント診断図・透過PNGは results_live/ に自動保存されます。", flush=True)
        print("=" * 65, flush=True)
        print("\n待機中... (新着CSV常時自動待機 / [Enter]: 最新CSV再解析, [S]: キャプチャ, [Q]: 終了)\n", flush=True)

        # フォルダ監視スレッドを起動 (バックグラウンドで新着CSVを常時自動検知)
        dir_thread = None
        if self.watch_dir and self.watch_dir.is_dir():
            dir_thread = threading.Thread(target=self._directory_watcher_loop, daemon=True)
            dir_thread.start()

        # Windowsの場合はctypesによるグローバルホットキー監視スレッドを起動
        hotkey_thread = None
        if sys.platform == "win32":
            hotkey_thread = threading.Thread(target=self._win32_hotkey_loop, daemon=True)
            hotkey_thread.start()

        gui_started = False
        if self.use_gui:
            try:
                from astrolaue.gui import AstroLaueHUD, TK_AVAILABLE
                if TK_AVAILABLE:
                    hud = AstroLaueHUD(
                        on_reanalyze_latest=self.trigger_restore_latest_csv,
                        on_capture_screen=self.trigger_capture_and_restore,
                        on_close=self.stop,
                        always_on_top=True,
                    )
                    # ウィンドウ作成を試行（DISPLAY環境変数がない等の場合はここでTclError）
                    hud._init_window()
                    self.gui = hud

                    # GUI初期化成功後にコンソール入力を別スレッドで待機
                    console_th = threading.Thread(target=self._console_input_loop, daemon=True)
                    console_th.start()
                    logger.info("最前面 GUI HUD を起動しました")
                    gui_started = True
                    hud.start_in_main_thread()
            except Exception as e_gui:
                logger.info("GUIの起動に失敗したため、コンソールモードで継続します: %s", e_gui)
                self.gui = None
                if not gui_started:
                    self._console_input_loop()
                else:
                    while self._running:
                        time.sleep(0.5)
                return

        if not gui_started:
            self._console_input_loop()

    def _console_input_loop(self) -> None:
        """コンソール入力待機ループ."""
        try:
            while self._running:
                raw = input()
                line = raw.strip().lower()
                if line in ("q", "quit", "exit"):
                    print("[*] 終了コマンドを受け付けました。常駐監視を終了します。", flush=True)
                    self.stop()
                    break
                elif line in ("c", "csv", "latest", "re", ""):
                    # 空Enter または [C] は最新CSVの再解析を実行 (誤って画面キャプチャを誤爆させない！)
                    print("[*] 最新CSVの再解析を実行します...", flush=True)
                    self.trigger_restore_latest_csv()
                elif line in ("s", "screen", "cap", "capture"):
                    # [S] で手動画面キャプチャ復元
                    self.trigger_capture_and_restore()
                elif line in ("a", "align", "alignment", "gonio", "axis"):
                    # 結晶主軸アライメントを実行
                    self.trigger_crystal_alignment()
                elif line in ("h", "help", "?"):
                    self._print_help()
                else:
                    print(f"[*] 不明なコマンドです: '{line}'. [H] でヘルプを表示します。", flush=True)
        except (KeyboardInterrupt, EOFError):
            print("\n[*] 常駐監視を終了します。", flush=True)
            self.stop()

    def _directory_watcher_loop(self) -> None:
        """測定PCの生データフォルダ (サブフォルダ含む) を常時監視し、新着CSVを検知したら即座に復元を実行します."""
        if not self.watch_dir or not self.watch_dir.is_dir():
            return

        # 起動時に既存のファイルを追跡登録 (サブフォルダ含む)
        try:
            existing = [f for f in self.watch_dir.rglob("*.[cC][sS][vV]") if f.is_file()]
            for f in existing:
                try:
                    self._seen_csv_files[str(f.resolve())] = f.stat().st_mtime
                except Exception:
                    pass
            logger.info("フォルダ監視初期化完了: %s (既存 %d ファイルを追跡)", self.watch_dir, len(self._seen_csv_files))
        except Exception as e_init:
            logger.warning("フォルダ監視初期化中に例外: %s", e_init)

        while self._running:
            try:
                if self.watch_dir.exists():
                    current_csvs = [f for f in self.watch_dir.rglob("*.[cC][sS][vV]") if f.is_file()]
                    new_candidates = []
                    for f in current_csvs:
                        f_key = str(f.resolve())
                        try:
                            mtime = f.stat().st_mtime
                            if f_key not in self._seen_csv_files or mtime > (self._seen_csv_files[f_key] + 1.0):
                                new_candidates.append((f, f_key, mtime))
                        except Exception:
                            pass

                    if new_candidates:
                        new_candidates.sort(key=lambda item: item[2])
                        for target_csv, f_key, mtime in new_candidates:
                            # 測定ソフトのファイル書き込み完了を待機 (ファイルサイズが安定するまで)
                            is_ready = False
                            for _ in range(5):
                                try:
                                    sz1 = target_csv.stat().st_size
                                    time.sleep(0.4)
                                    sz2 = target_csv.stat().st_size
                                    if sz1 == sz2 and sz1 > 1024:
                                        is_ready = True
                                        break
                                except Exception:
                                    time.sleep(0.4)

                            if is_ready:
                                self._seen_csv_files[f_key] = target_csv.stat().st_mtime
                                print(f"\n[*] 新着CSVを検知しました: {target_csv.name} (自動解析を開始します...)", flush=True)
                                if self.gui is not None:
                                    self.gui.update_status(f"📥 新着検知: {target_csv.name}", color="#00e5ff", filename=target_csv.name)
                                threading.Thread(
                                    target=self.trigger_restore_from_csv,
                                    args=(target_csv,),
                                    daemon=True,
                                ).start()
            except Exception as e_watch:
                logger.debug("フォルダ監視ループ中の例外: %s", e_watch)

            time.sleep(1.0)

    def _win32_hotkey_loop(self) -> None:
        """Windows API (RegisterHotKey) によるグローバルホットキー監視スレッド."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32

        MOD_ALT = 0x0001
        MOD_CONTROL = 0x0002
        MOD_SHIFT = 0x0004
        MOD_NOREPEAT = 0x4000
        VK_F9 = 0x78
        VK_F10 = 0x79
        VK_L = 0x4C
        VK_A = 0x41

        HOTKEY_ID_F9 = 101
        HOTKEY_ID_CTRL_SHIFT_L = 102
        HOTKEY_ID_F10 = 103
        HOTKEY_ID_CTRL_SHIFT_A = 104

        # ホットキー登録
        # 1. 復元用: F9 および Ctrl + Shift + L
        res1 = user32.RegisterHotKey(None, HOTKEY_ID_F9, MOD_NOREPEAT, VK_F9)
        if not res1:
            res1 = user32.RegisterHotKey(None, HOTKEY_ID_F9, 0, VK_F9)

        res2 = user32.RegisterHotKey(
            None, HOTKEY_ID_CTRL_SHIFT_L, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_L
        )

        # 2. アライメント用: F10 および Ctrl + Shift + A
        res3 = user32.RegisterHotKey(None, HOTKEY_ID_F10, MOD_NOREPEAT, VK_F10)
        if not res3:
            res3 = user32.RegisterHotKey(None, HOTKEY_ID_F10, 0, VK_F10)

        res4 = user32.RegisterHotKey(
            None, HOTKEY_ID_CTRL_SHIFT_A, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_A
        )

        if res1 or res2:
            logger.info("復元用グローバルホットキー [F9 / Ctrl+Shift+L] を登録しました")
        if res3 or res4:
            logger.info("アライメント用グローバルホットキー [F10 / Ctrl+Shift+A] を登録しました")

        WM_HOTKEY = 0x0312
        msg = wintypes.MSG()

        try:
            while self._running:
                # 100msごとにメッセージチェック (PeekMessageで非ブロッキング)
                PM_REMOVE = 0x0001
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY:
                        hk_id = msg.wParam
                        if hk_id in (HOTKEY_ID_F9, HOTKEY_ID_CTRL_SHIFT_L):
                            # 別スレッドで復元を実行
                            threading.Thread(
                                target=self.trigger_capture_and_restore, daemon=True
                            ).start()
                        elif hk_id in (HOTKEY_ID_F10, HOTKEY_ID_CTRL_SHIFT_A):
                            # 別スレッドで結晶アライメントを実行
                            threading.Thread(
                                target=self.trigger_crystal_alignment, daemon=True
                            ).start()

                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                else:
                    time.sleep(0.05)
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID_F9)
            user32.UnregisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_L)
            user32.UnregisterHotKey(None, HOTKEY_ID_F10)
            user32.UnregisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_A)
            logger.info("グローバルホットキーの登録を解除しました")
