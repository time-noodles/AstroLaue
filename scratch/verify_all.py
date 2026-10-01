"""260930-n (1〜5) および 260925 系列の包括的検証スクリプト."""

from pathlib import Path
import time
import numpy as np
import cv2
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig

def verify_dataset():
    out_base = Path("scratch/verify_results")
    out_base.mkdir(parents=True, exist_ok=True)

    test_images = [
        # 260930 系列 (課題対象)
        ("share/260930-1.png", "260930-1"),
        ("share/260930-2.png", "260930-2"),
        ("share/260930-3.png", "260930-3"),
        ("share/260930-4.png", "260930-4"),
        ("share/260930-5.png", "260930-5"),
        # 260925 系列 (回帰テスト)
        ("share/画像260925-1.png", "260925-1"),
        ("share/画像260925-2.png", "260925-2"),
        ("share/画像260925-3.png", "260925-3"),
        ("share/画像260925-4.png", "260925-4"),
        ("share/画像260925-5.png", "260925-5"),
        ("share/画像260925.png", "260925-0"),
    ]

    cfg = PipelineConfig(
        output_dir=out_base,
        num_iter=25,
        output_size=1024,
        n_r=384,
        n_theta=1080,
        show_diagnostic=True,
        save_intermediates=True,
        validate_physics=True,
        export_transparent=True,
        export_comparison=True,
    )

    print("=" * 80)
    print(f"{'Image':<12} | {'Spots':<6} | {'Low-Angle Spots':<16} | {'Avg FWHM Red':<13} | {'Flux Ratio':<11} | {'Status'}")
    print("=" * 80)

    for img_path_str, name in test_images:
        img_p = Path(img_path_str)
        if not img_p.is_file():
            print(f"Skipping {name}: file not found")
            continue

        item_out = out_base / name
        cfg.output_dir = item_out
        pipeline = AstroLauePipeline(cfg)

        t0 = time.time()
        res = pipeline.run(img_p)
        elapsed = time.time() - t0

        # 低角領域 (r0 <= r < 1.15 * r0) のスポット数
        low_angle_spots = [s for s in res.spots if res.r0 <= s.r < res.r0 * 1.15]

        # 先鋭化率
        reductions = [s.fwhm_reduction_ratio for s in res.spots if s.fwhm_before > 0]
        avg_red = float(np.mean(reductions)) if reductions else 0.0

        # フラックス比
        flux = res.physics_report.mean_flux_ratio if res.physics_report else 1.0

        # 透過PNGのチェック
        trans_p = item_out / "restored_transparent.png"
        trans_16_p = item_out / "restored_transparent_16bit.png"
        assert trans_p.is_file(), f"Missing {trans_p}"
        assert trans_16_p.is_file(), f"Missing {trans_16_p}"

        # 8-bit PNG の最小・最大輝度 (微弱ピークが可視化されているか)
        rgba = cv2.imread(str(trans_p), cv2.IMREAD_UNCHANGED)
        b, g, r, a = cv2.split(rgba)
        valid_b = b[a > 0]
        b_max = int(np.max(valid_b)) if len(valid_b) > 0 else 0
        b_p95 = int(np.percentile(valid_b, 95)) if len(valid_b) > 0 else 0

        status = f"OK ({elapsed:.1f}s, p95={b_p95})"
        print(f"{name:<12} | {len(res.spots):<6} | {len(low_angle_spots):<16} | {avg_red:<12.1f}% | {flux:<11.3f} | {status}")

    print("=" * 80)
    print("検証完了！")

if __name__ == "__main__":
    verify_dataset()
