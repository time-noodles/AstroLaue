"""AstroLaue 最前面 GUI HUD (gui.py) の単体テスト."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import tkinter as tk

from astrolaue.gui import AstroLaueHUD, TK_AVAILABLE
from astrolaue.alignment import AxisCandidate, LaueAlignmentConfig, LaueAlignmentResult
from astrolaue.exposure import ExposureOptimizationResult, SaturationStatus
from astrolaue.watcher import ResidentWatcher


def _create_mock_alignment_result() -> LaueAlignmentResult:
    """テスト用のアライメント結果オブジェクトを生成."""
    candidate = AxisCandidate(
        rank=1,
        u=0.02,
        v=-0.01,
        normal_vector=(0.02, -0.01, 0.999),
        delta_theta_x_deg=1.20,
        delta_theta_y_deg=-0.45,
        total_tilt_deg=1.35,
        score=15.2,
        contributing_zones=[1, 2],
        detector_x_mm=0.54,
        detector_y_mm=-0.27,
    )
    return LaueAlignmentResult(
        spots_xy_mm=np.zeros((5, 2)),
        intensities=np.zeros(5),
        normals=np.zeros((5, 3)),
        gnomonic_uv=np.zeros((5, 2)),
        zones=[],
        candidates=[candidate],
        best_candidate=candidate,
        camera_length_mm=27.0,
        config=LaueAlignmentConfig(),
    )


def _create_mock_exposure_result() -> ExposureOptimizationResult:
    """テスト用の露光時間最適化結果オブジェクトを生成."""
    return ExposureOptimizationResult(
        status=SaturationStatus.SATURATED,
        observed_max=7378767.0,
        estimated_true_max=10896874.0,
        background=500000.0,
        saturation_limit=7.72e6,
        target_max=6176000.0,
        recommended_ratio=0.567,
        peak_r_mm=15.8,
        peak_theta_deg=347.8,
        current_seconds=180.0,
        single_exposure_seconds=60.0,
        recommended_seconds=102.0,
        recommended_single_seconds=34.0,
        num_accumulations=3,
    )


def test_hud_initialization_and_always_on_top():
    """HUD ウィンドウの初期化と最前面属性の設定を検証."""
    if not TK_AVAILABLE:
        pytest.skip("Tkinter が利用できないためスキップ")

    try:
        hud = AstroLaueHUD(always_on_top=True)
        hud._init_window()
        assert hud.root is not None
        hud.root.update()

        # 最前面属性の検証 (Tkinter attributes("-topmost"))
        topmost = hud.root.attributes("-topmost")
        assert topmost in (1, True, "1")
        assert hud.always_on_top is True

        # 最前面トグル
        hud.set_always_on_top(False)
        assert hud.always_on_top is False

        hud.root.destroy()
    except tk.TclError:
        pytest.skip("GUI ディスプレイ (DISPLAY) が利用できないためスキップ")


def test_hud_update_alignment_and_exposure():
    """アライメントおよび露光最適化データの GUI 反映を検証."""
    if not TK_AVAILABLE:
        pytest.skip("Tkinter が利用できないためスキップ")

    try:
        hud = AstroLaueHUD(always_on_top=True)
        hud._init_window()
        assert hud.root is not None

        # 1. アライメント結果の反映
        align_mock = _create_mock_alignment_result()
        hud.update_alignment(align_mock, source_name="test_261011.CSV")
        hud.root.update()

        assert "+1.20°" in hud._lbl_dtheta_x.cget("text")
        assert "-0.45°" in hud._lbl_dtheta_y.cget("text")
        assert "1.35°" in hud._lbl_tilt.cget("text")
        assert "test_261011.CSV" in hud._lbl_file.cget("text")

        # 2. 露光最適化結果の反映
        exp_mock = _create_mock_exposure_result()
        hud.update_exposure(exp_mock)
        hud.root.update()

        assert "飽和検知" in hud._lbl_exp_status.cget("text")
        assert "60.0 秒" in hud._lbl_curr_time.cget("text")
        assert "34.0 秒" in hud._lbl_rec_time.cget("text")
        assert "57%" in hud._lbl_rec_ratio.cget("text")
        assert "7,378,767" in hud._lbl_max_intensity.cget("text")

        # 3. ステータス更新
        hud.update_status("✅ 復元完了", color="#69f0ae", filename="updated.CSV")
        hud.root.update()
        assert "✅ 復元完了" in hud._lbl_status.cget("text")
        assert "updated.CSV" in hud._lbl_file.cget("text")

        hud.root.destroy()
    except tk.TclError:
        pytest.skip("GUI ディスプレイ (DISPLAY) が利用できないためスキップ")


def test_hud_mocked_logic_and_formatting():
    """ディスプレイが無い環境(CI/ヘッドレス)でも動作するモックGUIロジック検証."""
    hud = AstroLaueHUD(always_on_top=True)

    # モック用UIコンポーネントを差し替え
    mock_root = MagicMock()
    # after(0, func) は即座に func() を実行するようにモック
    mock_root.after.side_effect = lambda delay, func=None, *args: func(*args) if func else None
    hud.root = mock_root

    hud._lbl_dtheta_x = MagicMock()
    hud._lbl_dtheta_y = MagicMock()
    hud._lbl_tilt = MagicMock()
    hud._lbl_center = MagicMock()
    hud._lbl_file = MagicMock()
    hud._lbl_status = MagicMock()
    hud._lbl_exp_status = MagicMock()
    hud._lbl_curr_time = MagicMock()
    hud._lbl_rec_time = MagicMock()
    hud._lbl_rec_ratio = MagicMock()
    hud._lbl_max_intensity = MagicMock()
    hud._lbl_advice = MagicMock()

    # 1. アライメント更新
    align_mock = _create_mock_alignment_result()
    hud.update_alignment(align_mock, source_name="sample.csv")
    hud._lbl_dtheta_x.config.assert_called_with(text="+1.20°")
    hud._lbl_dtheta_y.config.assert_called_with(text="-0.45°")
    hud._lbl_tilt.config.assert_called_with(text="全傾き角 ψ : 1.35°")

    # 2. 露光更新 (SATURATED)
    exp_mock = _create_mock_exposure_result()
    hud.update_exposure(exp_mock)
    hud._lbl_exp_status.config.assert_called_with(text="【⚠️ 飽和検知 / 上限到達】", fg="#ff5252")
    hud._lbl_rec_time.config.assert_called_with(text="34.0 秒", fg="#ff5252")

    # 3. 露光更新 (OPTIMAL)
    exp_optimal = ExposureOptimizationResult(
        status=SaturationStatus.OPTIMAL,
        observed_max=6000000.0,
        estimated_true_max=6000000.0,
        background=400000.0,
        saturation_limit=7.72e6,
        target_max=6176000.0,
        recommended_ratio=1.0,
        peak_r_mm=10.0,
        peak_theta_deg=45.0,
        current_seconds=60.0,
        single_exposure_seconds=60.0,
        recommended_seconds=60.0,
        recommended_single_seconds=60.0,
        num_accumulations=1,
    )
    hud.update_exposure(exp_optimal)
    hud._lbl_exp_status.config.assert_called_with(text="【✅ 最適露光 (目標範囲内)】", fg="#69f0ae")
    hud._lbl_rec_ratio.config.assert_called_with(
        text="設定推奨比率: 現行の 100% に現行維持を推奨",
        fg="#69f0ae",
    )

    # 4. クリップボードコピー
    hud._copy_angles()
    mock_root.clipboard_clear.assert_called_once()
    mock_root.clipboard_append.assert_called_once_with("Δθx = +1.20°, Δθy = -0.45°")


def test_hud_destroyed_window_exception_handling():
    """ウィンドウ破棄後に更新メソッドが呼ばれても例外でクラッシュしないことを検証."""
    hud = AstroLaueHUD(always_on_top=True)
    mock_root = MagicMock()
    mock_root.after.side_effect = tk.TclError('can\'t invoke "after" command: application has been destroyed')
    hud.root = mock_root

    # 例外が外に漏れず安全に握り潰されることを検証
    align_mock = _create_mock_alignment_result()
    hud.update_alignment(align_mock)

    exp_mock = _create_mock_exposure_result()
    hud.update_exposure(exp_mock)

    hud.update_status("test")


def test_resident_watcher_headless_fallback():
    """DISPLAYが無いヘッドレス環境で ResidentWatcher が安全にコンソールモードにフォールバックすることを検証."""
    watcher = ResidentWatcher(use_gui=True)

    # tk.Tk が TclError ("no display name") をスローするようにモック
    with patch("tkinter.Tk", side_effect=tk.TclError("no display name and no $DISPLAY environment variable")):
        with patch.object(watcher, "_console_input_loop", return_value=None) as mock_console:
            watcher.run()
            # GUI起動失敗時にフォールバックし、コンソール入力ループが呼び出されたことを検証
            assert mock_console.called
            assert watcher.gui is None


def test_resident_watcher_gui_integration():
    """ResidentWatcher の use_gui オプションおよびパラメータ設定を検証."""
    watcher = ResidentWatcher(use_gui=False)
    assert watcher.use_gui is False
    assert watcher.gui is None
