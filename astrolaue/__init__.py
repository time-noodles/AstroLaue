"""AstroLaue: 小型ラウエ画像復元パイプライン.

内蔵カメラの円周走査による方位角方向モーションブラーを
極座標変換と天体画像復元手法（Richardson-Lucy逆畳み込み等）によって
先鋭化・復元するパッケージ。
"""

from astrolaue.capture import extract_diffraction_roi, load_image, capture_window
from astrolaue.io import load_slaue_csv, SlaueRawData
from astrolaue.normalize import detect_center_and_radius, normalize_geometry, subtract_background
from astrolaue.polar import cartesian_to_polar, polar_to_cartesian
from astrolaue.deblur import (
    estimate_blur_length,
    create_psf_1d,
    richardson_lucy_circular,
    wiener_deblur_circular,
)
from astrolaue.analyze import detect_spots, measure_spot_fwhm, SpotResult
from astrolaue.validate import validate_physical_consistency, compute_residual_map, PhysicsValidationReport
from astrolaue.export import export_publication_rgba
from astrolaue.clipboard import copy_image_to_clipboard, copy_text_to_clipboard
from astrolaue.visualize import plot_diagnostic_figure, plot_batch_summary_figure, plot_comparison_figure
from astrolaue.pipeline import AstroLauePipeline, PipelineConfig, PipelineResult
from astrolaue.watcher import ResidentWatcher
from astrolaue.alignment import (
    LaueAlignmentConfig,
    ZoneLine,
    AxisCandidate,
    LaueAlignmentResult,
    extract_laue_spots_from_image,
    detector_to_normals,
    normals_to_gnomonic,
    detect_zone_lines,
    find_principal_axis_candidates,
    calculate_goniometer_angles,
    gnomonic_to_detector,
    zone_line_to_detector_hyperbola,
    solve_laue_alignment,
    plot_alignment_diagnostics,
    generate_synthetic_laue_sample,
)

__version__ = "0.2.0"

__all__ = [
    "extract_diffraction_roi",
    "load_image",
    "load_slaue_csv",
    "SlaueRawData",
    "capture_window",
    "detect_center_and_radius",
    "normalize_geometry",
    "subtract_background",
    "cartesian_to_polar",
    "polar_to_cartesian",
    "estimate_blur_length",
    "create_psf_1d",
    "richardson_lucy_circular",
    "wiener_deblur_circular",
    "detect_spots",
    "measure_spot_fwhm",
    "SpotResult",
    "validate_physical_consistency",
    "compute_residual_map",
    "PhysicsValidationReport",
    "export_publication_rgba",
    "copy_image_to_clipboard",
    "copy_text_to_clipboard",
    "plot_diagnostic_figure",
    "plot_batch_summary_figure",
    "plot_comparison_figure",
    "AstroLauePipeline",
    "PipelineConfig",
    "PipelineResult",
    "ResidentWatcher",
    "LaueAlignmentConfig",
    "ZoneLine",
    "AxisCandidate",
    "LaueAlignmentResult",
    "extract_laue_spots_from_image",
    "detector_to_normals",
    "normals_to_gnomonic",
    "detect_zone_lines",
    "find_principal_axis_candidates",
    "calculate_goniometer_angles",
    "gnomonic_to_detector",
    "zone_line_to_detector_hyperbola",
    "solve_laue_alignment",
    "plot_alignment_diagnostics",
    "generate_synthetic_laue_sample",
]

