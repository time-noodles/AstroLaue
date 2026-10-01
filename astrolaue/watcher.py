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
from typing import Optional

import cv2
import numpy as np

from astrolaue.capture import capture_window
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig

logger = logging.getLogger(__name__)


class ResidentWatcher:
    """ホットキー監視による常駐キャプチャ＆復元マネージャー."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        output_dir: str | Path = "./results_live",
        window_title: Optional[str] = None,
        monitor_index: int = 1,
        target_mode: str = "auto",
        hotkey_name: str = "F9",
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
        self._running = False
        self._is_processing = False
        self._lock = threading.Lock()

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
            print(f"  * 出力先 (最新)   : {self.output_dir.resolve() / 'restored_transparent.png'}")
            print(f"  * 出力先 (履歴)   : {self.history_dir.resolve() / f'{hist_prefix}.png'}")
            print("-" * 60)

            # 成功通知音 (Windows)
            self._notify_sound(success=True)

            print("\n>>> 次の測定を待機中... (F9 キー または Enter で実行, 'q' で終了) <<<")
            return True

        except Exception as ex:
            logger.error("キャプチャ＆復元中にエラーが発生しました: %s", ex, exc_info=True)
            print(f"\n[ERROR] 復元処理に失敗しました: {ex}")
            self._notify_sound(success=False)
            print("\n>>> 待機中... (F9 キー または Enter で再試行) <<<")
            return False

        finally:
            self._lock.release()

    def _notify_sound(self, success: bool = True) -> None:
        """復元完了をビープ音で通知します."""
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

    def run(self) -> None:
        """常駐監視ループを開始します."""
        self._running = True

        print("=" * 65)
        print("  AstroLaue: 常駐ホットキー監視モード (Resident Watcher)")
        print("=" * 65)
        print("  【トリガー操作】")
        print("    [F9]               : 現在の画面から即座にキャプチャ＆復元")
        print("    [Ctrl+Shift+L]     : 代替ホットキー (F9が使えない場合)")
        print("    [Enter] (コンソール): 手動トリガー")
        print("    [Q] + [Enter]      : 終了")
        print("  -------------------------------------------------------------")
        print("  * 特定のアプリに依存しません。")
        print("  * 測定ソフト、画像ビューア、ブラウザ等を画面に表示した状態で")
        print("    ホットキーを押すだけで自動復元されます。")
        print("  * 復元結果は results_live/ および results_live/history/ に自動保存されます。")
        print("=" * 65)
        print("\n待機中... (F9 キー または Enter を押してください)\n")

        # Windowsの場合はctypesによるグローバルホットキー監視スレッドを起動
        hotkey_thread = None
        if sys.platform == "win32":
            hotkey_thread = threading.Thread(target=self._win32_hotkey_loop, daemon=True)
            hotkey_thread.start()

        # メインスレッドではコンソール入力を待機
        try:
            while self._running:
                line = input()
                if line.strip().lower() in ("q", "quit", "exit"):
                    print("[*] 終了コマンドを受け付けました。常駐監視を終了します。")
                    self._running = False
                    break
                else:
                    # Enter押下で即時実行
                    self.trigger_capture_and_restore()
        except (KeyboardInterrupt, EOFError):
            print("\n[*] 常駐監視を終了します。")
            self._running = False

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
        VK_L = 0x4C

        HOTKEY_ID_F9 = 101
        HOTKEY_ID_CTRL_SHIFT_L = 102

        # ホットキー登録
        # 1. 単キー F9
        res1 = user32.RegisterHotKey(None, HOTKEY_ID_F9, MOD_NOREPEAT, VK_F9)
        if not res1:
            res1 = user32.RegisterHotKey(None, HOTKEY_ID_F9, 0, VK_F9)

        # 2. Ctrl + Shift + L
        res2 = user32.RegisterHotKey(
            None, HOTKEY_ID_CTRL_SHIFT_L, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_L
        )

        if res1:
            logger.info("グローバルホットキー [F9] を正常に登録しました")
        else:
            logger.warning("[F9] のホットキー登録に失敗しました (別アプリが占有している可能性があります)")

        if res2:
            logger.info("グローバルホットキー [Ctrl + Shift + L] を正常に登録しました")

        WM_HOTKEY = 0x0312
        msg = wintypes.MSG()

        try:
            while self._running:
                # 100msごとにメッセージチェック (PeekMessageで非ブロッキング)
                PM_REMOVE = 0x0001
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY:
                        # ホットキー押下
                        hk_id = msg.wParam
                        if hk_id in (HOTKEY_ID_F9, HOTKEY_ID_CTRL_SHIFT_L):
                            # 別スレッドで実行してメッセージループをブロックしない
                            exec_thread = threading.Thread(
                                target=self.trigger_capture_and_restore, daemon=True
                            )
                            exec_thread.start()
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                else:
                    time.sleep(0.05)
        finally:
            user32.UnregisterHotKey(None, HOTKEY_ID_F9)
            user32.UnregisterHotKey(None, HOTKEY_ID_CTRL_SHIFT_L)
            logger.info("グローバルホットキーの登録を解除しました")
