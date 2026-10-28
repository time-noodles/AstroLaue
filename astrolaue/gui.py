"""AstroLaue 最前面リアルタイム HUD ウィンドウ (astrolaue.gui).

測定PC画面（Rigaku Astray 等）の最前面に常にフロート表示され、
新着生データ (.CSV) または画面キャプチャから得られた
「結晶主軸アライメント (ゴニオ補正角)」と「推奨測定時間 (露光最適化)」を
リアルタイムに即座に表示します。
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from astrolaue.exposure import SaturationStatus

if TYPE_CHECKING:
    from astrolaue.alignment import LaueAlignmentResult
    from astrolaue.exposure import ExposureOptimizationResult

logger = logging.getLogger(__name__)

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except ImportError:
    TK_AVAILABLE = False


class AstroLaueHUD:
    """最前面に常駐フロート表示されるリアルタイム情報パネル (Always-on-Top HUD)."""

    def __init__(
        self,
        on_reanalyze_latest: Optional[Callable[[], Any]] = None,
        on_capture_screen: Optional[Callable[[], Any]] = None,
        on_close: Optional[Callable[[], Any]] = None,
        always_on_top: bool = True,
    ) -> None:
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter が利用できない環境です。")

        self.on_reanalyze_latest = on_reanalyze_latest
        self.on_capture_screen = on_capture_screen
        self.on_close_callback = on_close
        self.always_on_top = always_on_top

        self.root: Optional[tk.Tk] = None
        self._is_ready = threading.Event()
        self._last_angle_text = ""

        # UI コンポーネント保持用
        self._lbl_status: Any = None
        self._lbl_file: Any = None
        self._lbl_dtheta_x: Any = None
        self._lbl_dtheta_y: Any = None
        self._lbl_tilt: Any = None
        self._lbl_center: Any = None
        self._lbl_exp_status: Any = None
        self._lbl_curr_time: Any = None
        self._lbl_rec_time: Any = None
        self._lbl_rec_ratio: Any = None
        self._lbl_max_intensity: Any = None
        self._lbl_advice: Any = None
        self._topmost_var: Any = None

    def start_in_main_thread(self) -> None:
        """メインスレッドで GUI ループを起動します."""
        if self.root is None:
            self._init_window()
        self._is_ready.set()
        assert self.root is not None
        try:
            self.root.mainloop()
        finally:
            self._is_ready.clear()

    def _init_window(self) -> None:
        """Tkinter ウィンドウおよびウィジェットを構築します."""
        if self.root is not None:
            return
        root = tk.Tk()
        self.root = root
        root.title("AstroLaue Live HUD")
        root.configure(bg="#14141e")

        # ウィンドウサイズと初期位置 (画面右上)
        width, height = 440, 570
        try:
            screen_w = root.winfo_screenwidth()
            pos_x = max(20, screen_w - width - 40)
            pos_y = 40
            root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
        except Exception:
            root.geometry(f"{width}x{height}")

        root.minsize(380, 520)

        # 最前面設定
        self.set_always_on_top(self.always_on_top)

        # 閉じるイベントハンドラ
        root.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # スタイル設定
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        # === 1. ヘッダーフレーム ===
        header_frame = tk.Frame(root, bg="#1a1a28", pady=6, padx=10)
        header_frame.pack(fill="x", side="top")

        lbl_app = tk.Label(
            header_frame,
            text="✨ AstroLaue Live HUD",
            font=("Segoe UI", 12, "bold"),
            fg="#00e5ff",
            bg="#1a1a28",
        )
        lbl_app.pack(side="left")

        self._topmost_var = tk.BooleanVar(value=self.always_on_top)
        chk_topmost = tk.Checkbutton(
            header_frame,
            text="📌 最前面",
            variable=self._topmost_var,
            command=self._toggle_topmost,
            font=("Segoe UI", 9),
            fg="#e0e0e0",
            bg="#1a1a28",
            selectcolor="#252538",
            activebackground="#1a1a28",
            activeforeground="#00e5ff",
            relief="flat",
        )
        chk_topmost.pack(side="right")

        # === ステータスバー ===
        status_frame = tk.Frame(root, bg="#202030", padx=10, pady=3)
        status_frame.pack(fill="x", side="top")

        self._lbl_status = tk.Label(
            status_frame,
            text="🟢 新着生データ待機中...",
            font=("Segoe UI", 9, "bold"),
            fg="#69f0ae",
            bg="#202030",
        )
        self._lbl_status.pack(side="left")

        self._lbl_file = tk.Label(
            status_frame,
            text="未解析",
            font=("Segoe UI", 9),
            fg="#8888aa",
            bg="#202030",
        )
        self._lbl_file.pack(side="right")

        # === メインコンテナ ===
        container = tk.Frame(root, bg="#14141e", padx=10, pady=6)
        container.pack(fill="both", expand=True)

        # ==============================================================
        # カード 1: ⚡ 結晶主軸アライメント (Goniometer Angles)
        # ==============================================================
        card_align = tk.LabelFrame(
            container,
            text=" ⚡ 結晶主軸アライメント (推奨ゴニオ補正角) ",
            font=("Segoe UI", 10, "bold"),
            fg="#00e5ff",
            bg="#1c1c2b",
            relief="groove",
            bd=1,
            padx=10,
            pady=6,
        )
        card_align.pack(fill="x", pady=4)

        angles_row = tk.Frame(card_align, bg="#1c1c2b")
        angles_row.pack(fill="x", pady=2)

        # Δθx ボックス
        bx = tk.Frame(angles_row, bg="#26263a", padx=8, pady=4, relief="ridge", bd=1)
        bx.pack(side="left", expand=True, fill="both", padx=3)
        tk.Label(bx, text="Δθx (X軸回転)", font=("Segoe UI", 9), fg="#aaaabb", bg="#26263a").pack()
        self._lbl_dtheta_x = tk.Label(
            bx, text="--.--°", font=("Segoe UI", 20, "bold"), fg="#ffd700", bg="#26263a"
        )
        self._lbl_dtheta_x.pack()

        # Δθy ボックス
        by = tk.Frame(angles_row, bg="#26263a", padx=8, pady=4, relief="ridge", bd=1)
        by.pack(side="right", expand=True, fill="both", padx=3)
        tk.Label(by, text="Δθy (Y軸回転)", font=("Segoe UI", 9), fg="#aaaabb", bg="#26263a").pack()
        self._lbl_dtheta_y = tk.Label(
            by, text="--.--°", font=("Segoe UI", 20, "bold"), fg="#00e5ff", bg="#26263a"
        )
        self._lbl_dtheta_y.pack()

        # サブ詳細 (傾き角, 主軸座標)
        details_row = tk.Frame(card_align, bg="#1c1c2b")
        details_row.pack(fill="x", pady=4)

        self._lbl_tilt = tk.Label(
            details_row,
            text="全傾き角 ψ : --.--°",
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#1c1c2b",
        )
        self._lbl_tilt.pack(side="left")

        self._lbl_center = tk.Label(
            details_row,
            text="主軸 (001): (X=--.--, Y=--.--) mm",
            font=("Segoe UI", 9),
            fg="#8888aa",
            bg="#1c1c2b",
        )
        self._lbl_center.pack(side="right")

        # コピーボタン
        btn_copy = tk.Button(
            card_align,
            text="📋 補正角度をクリップボードにコピー",
            command=self._copy_angles,
            font=("Segoe UI", 9, "bold"),
            fg="#14141e",
            bg="#00e5ff",
            activebackground="#33eeff",
            relief="flat",
            pady=3,
            cursor="hand2",
        )
        btn_copy.pack(fill="x", pady=(4, 0))

        # ==============================================================
        # カード 2: ⏱️ 露光時間最適化 (Exposure Optimization)
        # ==============================================================
        card_exp = tk.LabelFrame(
            container,
            text=" ⏱️ 露光時間最適化 (80%目標) ",
            font=("Segoe UI", 10, "bold"),
            fg="#69f0ae",
            bg="#1c1c2b",
            relief="groove",
            bd=1,
            padx=10,
            pady=6,
        )
        card_exp.pack(fill="x", pady=6)

        # ステータスバッジ
        badge_row = tk.Frame(card_exp, bg="#1c1c2b")
        badge_row.pack(fill="x", pady=2)
        tk.Label(badge_row, text="露光判定 :", font=("Segoe UI", 9), fg="#aaaabb", bg="#1c1c2b").pack(side="left")
        self._lbl_exp_status = tk.Label(
            badge_row,
            text="【待機中】",
            font=("Segoe UI", 10, "bold"),
            fg="#8888aa",
            bg="#1c1c2b",
        )
        self._lbl_exp_status.pack(side="left", padx=6)

        # 時間表示ブロック (現行 -> 推奨)
        time_row = tk.Frame(card_exp, bg="#1c1c2b")
        time_row.pack(fill="x", pady=4)

        # 現行時間
        b_curr = tk.Frame(time_row, bg="#26263a", padx=6, pady=4, relief="ridge", bd=1)
        b_curr.pack(side="left", expand=True, fill="both", padx=2)
        tk.Label(b_curr, text="現行測定時間", font=("Segoe UI", 8), fg="#aaaabb", bg="#26263a").pack()
        self._lbl_curr_time = tk.Label(
            b_curr, text="--.- 秒", font=("Segoe UI", 14, "bold"), fg="#cccccc", bg="#26263a"
        )
        self._lbl_curr_time.pack()

        # 推奨時間
        b_rec = tk.Frame(time_row, bg="#26263a", padx=6, pady=4, relief="ridge", bd=1)
        b_rec.pack(side="right", expand=True, fill="both", padx=2)
        tk.Label(b_rec, text="推奨測定時間", font=("Segoe UI", 8, "bold"), fg="#69f0ae", bg="#26263a").pack()
        self._lbl_rec_time = tk.Label(
            b_rec, text="--.- 秒", font=("Segoe UI", 18, "bold"), fg="#69f0ae", bg="#26263a"
        )
        self._lbl_rec_time.pack()

        # 推奨比率 & 強度情報
        exp_detail = tk.Frame(card_exp, bg="#1c1c2b")
        exp_detail.pack(fill="x", pady=2)

        self._lbl_rec_ratio = tk.Label(
            exp_detail,
            text="設定推奨比率: 現行設定の -- %",
            font=("Segoe UI", 9, "bold"),
            fg="#ffd700",
            bg="#1c1c2b",
        )
        self._lbl_rec_ratio.pack(anchor="w")

        self._lbl_max_intensity = tk.Label(
            exp_detail,
            text="観測最大強度: -,---,--- counts (上限比 --.-%)",
            font=("Segoe UI", 8),
            fg="#8888aa",
            bg="#1c1c2b",
        )
        self._lbl_max_intensity.pack(anchor="w")

        self._lbl_advice = tk.Label(
            card_exp,
            text="改善効果: 生データを読み込むとアドバイスが表示されます",
            font=("Segoe UI", 8),
            fg="#aaaaaa",
            bg="#1c1c2b",
            wraplength=400,
            justify="left",
        )
        self._lbl_advice.pack(anchor="w", pady=(2, 0))

        # ==============================================================
        # クイックアクションボタン
        # ==============================================================
        btn_row = tk.Frame(container, bg="#14141e")
        btn_row.pack(fill="x", side="bottom", pady=4)

        btn_re = tk.Button(
            btn_row,
            text="🔄 最新CSV再解析",
            command=self._on_reanalyze_click,
            font=("Segoe UI", 9),
            fg="#ffffff",
            bg="#303045",
            activebackground="#454560",
            relief="flat",
            pady=4,
            cursor="hand2",
        )
        btn_re.pack(side="left", expand=True, fill="x", padx=2)

        btn_cap = tk.Button(
            btn_row,
            text="📸 画面キャプチャ復元",
            command=self._on_capture_click,
            font=("Segoe UI", 9),
            fg="#ffffff",
            bg="#303045",
            activebackground="#454560",
            relief="flat",
            pady=4,
            cursor="hand2",
        )
        btn_cap.pack(side="right", expand=True, fill="x", padx=2)

    def set_always_on_top(self, enable: bool) -> None:
        """最前面表示のオン/オフを設定します."""
        self.always_on_top = enable
        if self.root is not None:
            try:
                self.root.attributes("-topmost", enable)
            except Exception as e:
                logger.debug("最前面属性設定例外: %s", e)

    def _toggle_topmost(self) -> None:
        val = bool(self._topmost_var.get())
        self.set_always_on_top(val)

    def _on_window_close(self) -> None:
        """ウィンドウ終了ハンドラ."""
        if self.on_close_callback is not None:
            try:
                self.on_close_callback()
            except Exception:
                pass
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _copy_angles(self) -> None:
        """クリップボードに補正角度をコピー."""
        if not self._last_angle_text or self.root is None:
            return

        # 1. OSクリップボード連携
        try:
            from astrolaue.clipboard import copy_text_to_clipboard
            copy_text_to_clipboard(self._last_angle_text)
        except Exception:
            pass

        # 2. Tkinter クリップボード更新
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(self._last_angle_text)
            self.root.update_idletasks()
            if self._lbl_status is not None:
                orig_text = self._lbl_status.cget("text")
                self._lbl_status.config(text="📋 補正角をクリップボードにコピーしました！", fg="#00e5ff")

                def _restore_status() -> None:
                    try:
                        if self.root is not None and self._lbl_status is not None:
                            self._lbl_status.config(text=orig_text, fg="#69f0ae")
                    except Exception:
                        pass

                try:
                    self.root.after(2000, _restore_status)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("クリップボードコピー例外: %s", e)

    def _on_reanalyze_click(self) -> None:
        if self.on_reanalyze_latest is not None:
            threading.Thread(target=self.on_reanalyze_latest, daemon=True).start()

    def _on_capture_click(self) -> None:
        if self.on_capture_screen is not None:
            threading.Thread(target=self.on_capture_screen, daemon=True).start()

    # ==============================================================
    # スレッドセーフな情報更新 API
    # ==============================================================

    def update_alignment(self, align_res: LaueAlignmentResult, source_name: str = "") -> None:
        """アライメント結果を最前面GUIに反映します (スレッドセーフ)."""
        if self.root is None:
            return

        def _apply() -> None:
            try:
                best = align_res.best_candidate
                self._lbl_dtheta_x.config(text=f"{best.delta_theta_x_deg:+.2f}°")
                self._lbl_dtheta_y.config(text=f"{best.delta_theta_y_deg:+.2f}°")
                self._lbl_tilt.config(text=f"全傾き角 ψ : {best.total_tilt_deg:.2f}°")
                self._lbl_center.config(text=f"主軸 (001): (X={best.detector_x_mm:+.2f}, Y={best.detector_y_mm:+.2f}) mm")

                self._last_angle_text = f"Δθx = {best.delta_theta_x_deg:+.2f}°, Δθy = {best.delta_theta_y_deg:+.2f}°"
                if source_name:
                    self._lbl_file.config(text=source_name)
                self._lbl_status.config(text="⚡ アライメント速報取得完了", fg="#00e5ff")
            except Exception as e:
                logger.debug("GUI アライメント反映例外: %s", e)

        try:
            self.root.after(0, _apply)
        except Exception as e:
            logger.debug("GUI アライメント反映例外 (ウィンドウ破棄済み等): %s", e)

    def update_exposure(self, exp_res: ExposureOptimizationResult) -> None:
        """露光時間最適化結果を最前面GUIに反映します (スレッドセーフ)."""
        if self.root is None:
            return

        def _apply() -> None:
            try:
                # ステータスバッジ
                st = exp_res.status
                denom = max(exp_res.saturation_limit, 1.0)
                ratio_sat = (exp_res.observed_max / denom) * 100.0

                if st == SaturationStatus.SATURATED:
                    self._lbl_exp_status.config(text="【⚠️ 飽和検知 / 上限到達】", fg="#ff5252")
                    adv_color = "#ff5252"
                    rec_color = "#ff5252"
                elif st == SaturationStatus.SLIGHTLY_HIGH:
                    self._lbl_exp_status.config(text="【⚠️ やや高め / 飽和寸前】", fg="#ffd600")
                    adv_color = "#ffd600"
                    rec_color = "#ffd600"
                elif st == SaturationStatus.OPTIMAL:
                    self._lbl_exp_status.config(text="【✅ 最適露光 (目標範囲内)】", fg="#69f0ae")
                    adv_color = "#69f0ae"
                    rec_color = "#69f0ae"
                else:
                    self._lbl_exp_status.config(text="【ℹ️ 露光不足】", fg="#40c4ff")
                    adv_color = "#40c4ff"
                    rec_color = "#40c4ff"

                # 現行時間
                curr_sec = exp_res.single_exposure_seconds if exp_res.single_exposure_seconds is not None else exp_res.current_seconds
                if curr_sec is not None:
                    self._lbl_curr_time.config(text=f"{curr_sec:.1f} 秒")
                else:
                    self._lbl_curr_time.config(text="不明")

                # 推奨時間
                rec_sec = exp_res.recommended_single_seconds if exp_res.recommended_single_seconds is not None else exp_res.recommended_seconds
                if rec_sec is not None:
                    self._lbl_rec_time.config(text=f"{rec_sec:.1f} 秒", fg=rec_color)
                else:
                    self._lbl_rec_time.config(text="--.- 秒", fg="#8888aa")

                # 推奨比率
                pct = exp_res.recommended_ratio * 100.0
                if st in (SaturationStatus.SATURATED, SaturationStatus.SLIGHTLY_HIGH):
                    rec_badge = "短縮設定を推奨"
                elif st == SaturationStatus.OPTIMAL:
                    rec_badge = "現行維持を推奨"
                else:
                    rec_badge = "延長を推奨"

                self._lbl_rec_ratio.config(
                    text=f"設定推奨比率: 現行の {pct:.0f}% に{rec_badge}",
                    fg=adv_color,
                )

                # 最大強度
                true_str = f" / 真値外挿: {exp_res.estimated_true_max:,.0f}" if exp_res.estimated_true_max > exp_res.observed_max else ""
                self._lbl_max_intensity.config(
                    text=f"観測最大: {exp_res.observed_max:,.0f} counts (上限比 {ratio_sat:.1f}%){true_str}"
                )

                # 改善アドバイス
                if st == SaturationStatus.SATURATED:
                    self._lbl_advice.config(text="改善効果: 1回ごとの物理飽和とピーク頭打ちを防ぎ、デブラー先鋭化を極限まで発揮")
                elif st == SaturationStatus.OPTIMAL:
                    self._lbl_advice.config(text="判定: 現在の照射条件で最高のデブラー復元精度が得られています")
                elif st == SaturationStatus.UNDER_EXPOSED:
                    self._lbl_advice.config(text="改善効果: 1回あたりの照射時間を延長することで微弱ピークのS/N比が向上し、検出感度が向上")
                else:
                    self._lbl_advice.config(text="改善効果: 飽和マージンを確保し、測定強度の線形性とデブラー復元精度を担保")

            except Exception as e:
                logger.debug("GUI 露光反映例外: %s", e)

        try:
            self.root.after(0, _apply)
        except Exception as e:
            logger.debug("GUI 露光反映例外 (ウィンドウ破棄済み等): %s", e)

    def update_status(self, text: str, color: str = "#69f0ae", filename: Optional[str] = None) -> None:
        """ステータスメッセージを更新します (スレッドセーフ)."""
        if self.root is None:
            return

        def _apply() -> None:
            try:
                if self._lbl_status is not None:
                    self._lbl_status.config(text=text, fg=color)
                if filename is not None and self._lbl_file is not None:
                    self._lbl_file.config(text=filename)
            except Exception as e:
                logger.debug("GUI ステータス反映例外: %s", e)

        try:
            self.root.after(0, _apply)
        except Exception as e:
            logger.debug("GUI ステータス反映例外 (ウィンドウ破棄済み等): %s", e)
