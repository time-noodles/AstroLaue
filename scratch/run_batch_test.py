"""バッチ処理の一括実行テストスクリプト.

全 11 画像に対して AstroLauePipeline.run_batch('share') を実行し、
summary_report.csv および batch_summary_plot.png を生成して検証します。
"""

import sys
import logging
from pathlib import Path
import pandas as pd

from astrolaue.pipeline import AstroLauePipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_test")

def main():
    share_dir = Path("share")
    if not share_dir.exists():
        logger.error("share ディレクトリが見つかりません。")
        sys.exit(1)

    out_dir = Path("results_batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        output_dir=out_dir,
        num_iter=25,
        bg_method="tophat",
        bg_kernel_size=51,
        r_inner_ratio=1.035,
        r_outer_ratio=0.995,
        export_asinh=True,
        show_diagnostic=True,
        save_intermediates=True,
        validate_physics=True,
        export_transparent=True,
    )

    pipeline = AstroLauePipeline(config)
    logger.info("バッチ処理を開始します (対象: %s)...", share_dir)
    csv_path = pipeline.run_batch(share_dir)
    logger.info("バッチ処理完了! CSV: %s", csv_path)

    # 結果の検証
    df = pd.read_csv(csv_path)
    print("\n" + "=" * 80)
    print("AstroLaue Batch Processing Verification Results")
    print("=" * 80)
    print(df[["image_name", "status", "num_spots", "avg_fwhm_reduction_pct", "mean_flux_ratio", "time_sec"]].to_string(index=False))
    print("=" * 80)

    # 成果物の存在確認
    plot_path = out_dir / "batch_summary_plot.png"
    assert csv_path.exists(), f"CSVが存在しません: {csv_path}"
    assert plot_path.exists(), f"一覧比較図が存在しません: {plot_path}"
    print(f"\n[OK] summary_report.csv size: {csv_path.stat().st_size} bytes")
    print(f"[OK] batch_summary_plot.png size: {plot_path.stat().st_size} bytes")

    # 全画像の成功を確認
    success_count = (df["status"] == "SUCCESS").sum()
    total_count = len(df)
    print(f"[OK] 全 {total_count} 枚中 {success_count} 枚が成功しました。")
    if success_count < total_count:
        print("[WARNING] 一部画像で失敗がありました。エラーメッセージを確認してください。")

if __name__ == "__main__":
    main()
