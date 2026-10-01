"""シミュレーション検証実行スクリプト."""

import sys
from pathlib import Path
import cv2

from astrolaue.simulation import (
    generate_realistic_laue_simulation,
    evaluate_peak_recovery,
    plot_recovery_benchmark_report,
    plot_distribution_comparison,
)
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig

def main():
    out_dir = Path("results_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("1. リアルな模擬ラウエ像（晶帯軸・非対称走査ブラー・散乱ノイズ）を生成中...")
    blurred_img, gt_img, gt_spots = generate_realistic_laue_simulation(
        size=512,
        center=(256.0, 256.0),
        r0=28.5,
        r_outer=232.0,
        blur_length_pix=12.5,
        num_zones=12,
        spots_per_zone=9,
        spot_sigma=0.9,
        noise_sigma=0.012,
        random_seed=42,
    )

    # 入力画像としてPNG保存
    sim_path = out_dir / "simulated_input.png"
    cv2.imwrite(str(sim_path), (blurred_img * 255.0).astype("uint8"))
    cv2.imwrite(str(out_dir / "ground_truth.png"), (gt_img * 255.0).astype("uint8"))
    print(f"   生成完了: スポット数={len(gt_spots)} 個, 画像サイズ={blurred_img.shape}")

    print("2. AstroLauePipeline による模擬像の復元・解析を実行中...")
    cfg = PipelineConfig(
        output_dir=out_dir / "pipe_out",
        num_iter=28,
        output_size=512,
        n_r=384,
        n_theta=1080,
        spot_max_spots=200,
        show_diagnostic=True,
        save_intermediates=True,
    )
    pipe = AstroLauePipeline(cfg)
    result = pipe.run(sim_path)
    print(f"   模擬像パイプライン完了: 検出スポット数={len(result.spots)} 個")

    print("3. ピーク再現性（位置・強度・先鋭化・F1）のベンチマーク評価中...")
    metrics, matched_pairs = evaluate_peak_recovery(
        pipeline_result=result,
        ground_truth_spots=gt_spots,
        ground_truth_image=gt_img,
        match_distance_tolerance_px=4.5,
    )

    print("\n" + "=" * 60)
    print("AstroLaue Peak Recovery Benchmark Results")
    print("=" * 60)
    print(f"• Ground Truth Spots  : {metrics.num_ground_truth}")
    print(f"• Detected Spots      : {metrics.num_detected}")
    print(f"• Matched Spots       : {metrics.num_matched}")
    print(f"• Recall (再現率)     : {metrics.recall*100:.1f} %")
    print(f"• Precision (適合率)  : {metrics.precision*100:.1f} %")
    print(f"• F1-Score (調和平均) : {metrics.f1_score:.3f}")
    print(f"• Position RMSE       : {metrics.position_rmse_px:.2f} px")
    print(f"• Mean Position Error : {metrics.mean_position_error_px:.2f} px")
    print(f"• Intensity Linearity : R² = {metrics.intensity_r2:.3f}")
    print(f"• Flux Recovery Ratio : {metrics.mean_flux_recovery_ratio:.3f}")
    print(f"• Avg FWHM Reduction  : {metrics.avg_fwhm_reduction_pct:.1f} %")
    print(f"• Image Quality       : {metrics.psnr_db:.1f} dB")
    print("=" * 60)

    # 4. レポート図の保存
    report_path = out_dir / "synthetic_recovery_benchmark.png"
    plot_recovery_benchmark_report(
        ground_truth_image=gt_img,
        blurred_image=blurred_img,
        pipeline_result=result,
        ground_truth_spots=gt_spots,
        matched_pairs=matched_pairs,
        metrics=metrics,
        output_path=report_path,
    )
    print(f"\n[OK] ベンチマークレポート図を保存しました: {report_path}")

    # 5. 実機データ (/share/260930-1.png) との分布比較図を出力
    real_path = Path("share/260930-1.png")
    if real_path.exists():
        print("5. 実機データ (share/260930-1.png) の解析および非対称PSF・コントラスト最適化を生成中...")
        real_cfg = PipelineConfig(
            output_dir=out_dir / "pipe_real",
            kernel_type="auto",
            export_linear=True,
            export_contrast_optimized=True,
            num_iter=28,
            output_size=512,
            n_r=384,
            n_theta=1080,
            spot_max_spots=200,
            show_diagnostic=True,
            save_intermediates=True,
        )
        real_pipe = AstroLauePipeline(real_cfg)
        real_result = real_pipe.run(real_path)

        comp_path = out_dir / "distribution_comparison.png"
        plot_distribution_comparison(
            real_pipeline_result=real_result,
            sim_pipeline_result=result,
            output_path=comp_path,
        )
        print(f"[OK] 実機 vs シミュレーション分布比較グラフを保存しました: {comp_path}")
        print(f"[OK] 実機 非対称同定PSF: {real_result.psf_params}")
        print(f"[OK] 実機 コントラスト最適化画像: {real_result.contrast_optimized_png_path}")
        print(f"[OK] 実機 3並列比較図: {real_result.contrast_comparison_path}")
    else:
        print(f"[WARN] 実機画像が見つかりません: {real_path}")

if __name__ == "__main__":
    main()
