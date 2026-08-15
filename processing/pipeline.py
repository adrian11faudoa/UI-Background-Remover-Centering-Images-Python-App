"""
Image processing pipeline for ClothingSnap.
Handles background removal, alpha cropping, centering, and edge refinement.
"""
import os
import io
import logging
import tempfile
import traceback
import threading
import time
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable, Optional, Tuple, TypeVar

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

BASE_SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
HEIF_FORMATS = {".heic", ".heif"}
SUPPORTED_FORMATS = BASE_SUPPORTED_FORMATS | HEIF_FORMATS


def _enable_optional_pillow_decoders() -> None:
    """
    Enable extra Pillow decoders when optional packages are installed.
    """
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        logger.info("Enabled HEIC/HEIF input support.")
    except ImportError:
        logger.debug("pillow-heif not installed; HEIC/HEIF input support disabled.")
    except Exception as exc:
        logger.warning("Could not enable HEIC/HEIF input support: %s", exc)


_enable_optional_pillow_decoders()

_REMBG_SESSIONS = {}
_REMBG_LOCK = threading.Lock()
T = TypeVar("T")


@dataclass
class ProcessingConfig:
    canvas_width: int = 1200
    canvas_height: int = 1600
    fill_ratio: float = 1.0
    padding_ratio: float = 0.0
    output_format: str = "WEBP"
    rembg_model: str = "isnet-general-use"
    webp_quality: int = 90
    webp_lossless: bool = False
    smooth_edges: bool = True
    edge_smooth_radius: int = 1
    alpha_threshold: int = 2
    model_timeout_sec: int = 60
    remove_timeout_sec: int = 180
    use_cropped_size_canvas: bool = True
    auto_recover_mask: bool = True
    secondary_rembg_model: Optional[str] = "u2net"
    fragment_keep_ratio: float = 0.05
    hole_recovery_trigger_ratio: float = 0.015
    hole_fill_max_ratio: float = 0.03
    hole_fill_small_ratio: float = 0.004
    hole_fill_min_rel_y: float = 0.18
    min_component_area_ratio: float = 0.00003
    suspicious_component_ratio: float = 0.05
    suspicious_min_solidity: float = 0.34
    reconstruction_close_ratio: float = 0.055
    debug_output_dir: Optional[str] = None


@dataclass
class MaskValidation:
    is_suspicious: bool
    reason: str = ""
    component_count: int = 0
    large_component_count: int = 0
    foreground_ratio: float = 0.0
    bbox_solidity: float = 0.0
    enclosed_hole_ratio: float = 0.0


def _run_with_timeout(operation: Callable[[], T], timeout_sec: int, label: str) -> T:
    """Run an operation in a worker thread and fail fast if it blocks too long."""
    result = {}
    error = {}

    def _target():
        try:
            result["value"] = operation()
        except Exception as exc:
            error["value"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout_sec)

    if worker.is_alive():
        raise TimeoutError(f"{label} timed out after {timeout_sec}s")
    if "value" in error:
        raise error["value"]
    return result["value"]


def _load_rembg_session(model_name: str):
    from rembg import new_session

    logger.info("[startup] Loading rembg session (%s)...", model_name)
    start = time.perf_counter()
    session = new_session(model_name)
    logger.info("[startup] rembg session (%s) loaded in %.2fs", model_name, time.perf_counter() - start)
    return session


def get_rembg_session(timeout_sec: int = 60, model_name: str = "isnet-general-use"):
    """Get a cached rembg session to avoid reloading the model per image."""
    if model_name in _REMBG_SESSIONS:
        return _REMBG_SESSIONS[model_name]

    with _REMBG_LOCK:
        if model_name not in _REMBG_SESSIONS:
            _REMBG_SESSIONS[model_name] = _run_with_timeout(
                lambda: _load_rembg_session(model_name),
                timeout_sec,
                f"Model loading ({model_name})",
            )
    return _REMBG_SESSIONS[model_name]


def warmup_background_model(timeout_sec: int = 60, model_name: str = "isnet-general-use") -> bool:
    """Pre-load the background model once to prevent repeated blocking during batch runs."""
    try:
        get_rembg_session(timeout_sec=timeout_sec, model_name=model_name)
        return True
    except ImportError:
        raise RuntimeError(
            "rembg is not installed. Run: pip install rembg[gpu] or pip install rembg"
        )
    except Exception as exc:
        logger.error("Model warmup failed: %s", exc)
        return False


def remove_background(image: Image.Image, config: ProcessingConfig) -> Image.Image:
    """
    Remove background using rembg with a cached session.
    This avoids repeated model loads and gives timeout diagnostics.
    """
    try:
        from rembg import remove
    except ImportError as exc:
        raise RuntimeError(
            "rembg is not installed. Run: pip install rembg[gpu] or pip install rembg"
        ) from exc

    session = get_rembg_session(timeout_sec=config.model_timeout_sec, model_name=config.rembg_model)

    def _do_remove():
        return remove(
            image,
            session=session,
            alpha_matting=False,
            post_process_mask=False,
        )

    start = time.perf_counter()
    result = _run_with_timeout(_do_remove, config.remove_timeout_sec, "Background removal")
    logger.info("Background removed in %.2fs", time.perf_counter() - start)
    if result.mode != "RGBA":
        result = result.convert("RGBA")
    return result


def refine_alpha_edges(image: Image.Image, smooth_radius: int = 1) -> Image.Image:
    """
    Clean alpha mask:
    - remove tiny noise
    - smooth edges
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    r, g, b, a = image.split()
    a_np = np.array(a, dtype=np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    a_np = cv2.morphologyEx(a_np, cv2.MORPH_OPEN, kernel, iterations=1)
    a_np = cv2.morphologyEx(a_np, cv2.MORPH_CLOSE, kernel, iterations=1)

    if smooth_radius > 0:
        blur_size = 2 * smooth_radius + 1
        a_np = cv2.GaussianBlur(a_np, (blur_size, blur_size), 0)

    # Drop near-zero alpha leftovers to keep transparent background clean.
    a_np[a_np < 2] = 0

    a_clean = Image.fromarray(a_np, mode="L")
    return Image.merge("RGBA", (r, g, b, a_clean))


def _save_debug_image(config: ProcessingConfig, image_name: str, stage: str, image: Image.Image) -> None:
    """Save a pipeline stage image when debug output is explicitly enabled."""
    if not config.debug_output_dir:
        return

    try:
        debug_dir = Path(config.debug_output_dir) / Path(image_name).stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        image.save(debug_dir / f"debug_{stage}.png")
    except Exception as exc:
        logger.warning("Could not save debug image %s/%s: %s", image_name, stage, exc)


def _alpha_mask_image(mask: np.ndarray) -> Image.Image:
    """Create a grayscale debug image from a boolean or uint8 mask."""
    if mask.dtype == bool:
        return Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def _remove_tiny_components(
    binary_mask: np.ndarray,
    min_area: int,
) -> Optional[np.ndarray]:
    """Remove isolated specks without assuming the garment is a single component."""
    binary_u8 = binary_mask.astype(np.uint8)
    if not binary_u8.any():
        return None

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if labels_count <= 1:
        return binary_u8.astype(bool)

    keep = np.zeros_like(binary_mask, dtype=bool)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep |= labels == label

    if keep.any():
        return keep

    areas = stats[1:, cv2.CC_STAT_AREA]
    return labels == (int(np.argmax(areas)) + 1)


def _foreground_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return a tight foreground bbox from a boolean mask."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def validate_garment_mask(mask: np.ndarray, config: ProcessingConfig) -> MaskValidation:
    """
    Detect masks that look structurally unsafe for clothing.
    This intentionally checks topology and area, not foreground color.
    """
    if not mask.any():
        return MaskValidation(True, "empty alpha mask")

    h, w = mask.shape
    img_area = float(max(1, h * w))
    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    component_count = max(0, labels_count - 1)
    areas = stats[1:, cv2.CC_STAT_AREA] if labels_count > 1 else np.array([], dtype=np.int32)
    largest = int(areas.max()) if areas.size else 0
    large_min = max(128, int(largest * config.suspicious_component_ratio))
    large_count = int(np.sum(areas >= large_min)) if largest else 0

    bbox = _foreground_bbox_from_mask(mask)
    if bbox is None:
        return MaskValidation(True, "empty alpha mask")

    left, top, right, bottom = bbox
    bbox_area = float(max(1, (right - left) * (bottom - top)))
    fg_area = float(np.count_nonzero(mask))
    bbox_solidity = fg_area / bbox_area
    hole_ratio = _estimate_enclosed_hole_ratio(mask)
    fg_ratio = fg_area / img_area

    reasons = []
    fragmented = large_count >= 2
    sparse = bbox_solidity < config.suspicious_min_solidity and bbox_area / img_area > 0.05

    if fragmented:
        reasons.append("multiple substantial foreground components")
    if hole_ratio >= config.hole_recovery_trigger_ratio and (fragmented or sparse):
        reasons.append("large enclosed transparent region")
    if sparse:
        reasons.append("foreground too sparse inside object bounds")
    if fg_ratio < 0.01:
        reasons.append("foreground area abnormally small")

    return MaskValidation(
        is_suspicious=bool(reasons),
        reason=", ".join(reasons),
        component_count=component_count,
        large_component_count=large_count,
        foreground_ratio=fg_ratio,
        bbox_solidity=bbox_solidity,
        enclosed_hole_ratio=hole_ratio,
    )


def _close_mask_for_garment_reconstruction(mask: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Bridge likely garment regions when the segmentation split the same product."""
    bbox = _foreground_bbox_from_mask(mask)
    if bbox is None:
        return mask

    left, top, right, bottom = bbox
    span = max(1, min(right - left, bottom - top))
    k = max(5, int(round(span * config.reconstruction_close_ratio)))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel, iterations=2) > 0

    points = np.column_stack(np.where(mask > 0))
    if len(points) >= 3:
        hull_points = np.flip(points, axis=1).astype(np.int32)
        hull = cv2.convexHull(hull_points)
        hull_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillConvexPoly(hull_mask, hull, 255)
        hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_OPEN, kernel, iterations=1) > 0
        closed |= hull_mask

    return closed


def _restore_legitimate_openings(original_mask: np.ndarray, reconstructed_mask: np.ndarray) -> np.ndarray:
    """
    Keep large top openings transparent after reconstruction.
    Missing fabric in the torso is usually central/lower; neck openings are
    high in the product bbox and should remain transparent.
    """
    bbox = _foreground_bbox_from_mask(reconstructed_mask)
    if bbox is None:
        return reconstructed_mask

    left, top, right, bottom = bbox
    bbox_h = max(1, bottom - top)
    bbox_area = float(max(1, (right - left) * bbox_h))
    added = reconstructed_mask & ~original_mask

    inverse = (~original_mask & reconstructed_mask).astype(np.uint8)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    restored = reconstructed_mask.copy()

    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = stats[label]
        if area <= 0:
            continue

        _, cy = centroids[label]
        rel_y = (float(cy) - top) / float(bbox_h)
        area_ratio = float(area) / bbox_area
        touches_top_zone = y <= top + int(bbox_h * 0.22)
        wide_top_opening = touches_top_zone and comp_w >= max(8, int((right - left) * 0.08))

        if wide_top_opening and rel_y <= 0.35 and area_ratio >= 0.003:
            restored[labels == label] = False

    return restored | (original_mask & ~added)


def reconstruct_garment_mask(mask: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """
    Recover likely missing clothing regions without color-thresholding.
    The reconstruction only runs for suspicious masks and operates on foreground
    topology: bridge nearby product parts, fill internal segmentation dropouts,
    then keep plausible neck/opening areas transparent.
    """
    if not mask.any():
        return mask

    reconstructed = _close_mask_for_garment_reconstruction(mask, config)
    reconstructed = _fill_enclosed_holes(
        reconstructed,
        max_hole_ratio=0.40,
        small_hole_ratio=max(config.hole_fill_small_ratio, 0.006),
        min_rel_y=config.hole_fill_min_rel_y,
    )
    reconstructed = _restore_legitimate_openings(mask, reconstructed)

    min_area = max(32, int(mask.size * config.min_component_area_ratio))
    cleaned = _remove_tiny_components(reconstructed, min_area=min_area)
    return cleaned if cleaned is not None else mask


def build_refined_alpha(
    source_image: Image.Image,
    segmented_image: Image.Image,
    config: ProcessingConfig,
) -> Tuple[np.ndarray, MaskValidation]:
    """
    Convert model alpha to a garment-aware alpha channel.
    This is the central postprocessing step used before crop/center/export.
    """
    source_rgba = np.array(source_image.convert("RGBA"), dtype=np.uint8)
    segmented_rgba = np.array(segmented_image.convert("RGBA"), dtype=np.uint8)
    alpha = segmented_rgba[:, :, 3]
    raw_mask = alpha > config.alpha_threshold
    min_area = max(32, int(raw_mask.size * config.min_component_area_ratio))

    cleaned_mask = _remove_tiny_components(raw_mask, min_area=min_area)
    if cleaned_mask is None:
        return np.zeros_like(alpha, dtype=np.uint8), validate_garment_mask(raw_mask, config)

    validation = validate_garment_mask(cleaned_mask, config)
    if validation.is_suspicious:
        logger.info("Mask flagged as suspicious: %s", validation.reason)
        reconstructed_mask = reconstruct_garment_mask(cleaned_mask, config)
        reconstructed_validation = validate_garment_mask(reconstructed_mask, config)
        if (
            np.count_nonzero(reconstructed_mask) >= np.count_nonzero(cleaned_mask)
            and reconstructed_validation.bbox_solidity >= validation.bbox_solidity
        ):
            cleaned_mask = reconstructed_mask
            validation = reconstructed_validation

    added_by_reconstruction = cleaned_mask & ~raw_mask
    refined_alpha = np.where(cleaned_mask, alpha, 0).astype(np.uint8)
    refined_alpha[added_by_reconstruction] = 255

    source_alpha = source_rgba[:, :, 3]
    if np.any(source_alpha < 255):
        refined_alpha = np.minimum(refined_alpha, source_alpha)

    refined_alpha[refined_alpha <= config.alpha_threshold] = 0
    return refined_alpha, validation


def _best_component_mask(binary: np.ndarray) -> Optional[np.ndarray]:
    """
    Pick the most likely clothing component:
    - prefers larger components
    - prefers center-near components
    - penalizes border-touching components
    """
    binary_u8 = binary.astype(np.uint8)
    if not binary_u8.any():
        return None

    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if labels_count <= 1:
        return binary_u8.astype(bool)

    h, w = binary_u8.shape
    img_area = float(h * w)
    center_x = (w - 1) * 0.5
    center_y = (h - 1) * 0.5

    border_labels = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_hits = np.bincount(border_labels, minlength=labels_count)

    min_area = max(64, int(img_area * 0.00005))
    best_label = 0
    best_score = -1.0

    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cx, cy = centroids[label]
        dx = (cx - center_x) / max(1.0, center_x)
        dy = (cy - center_y) / max(1.0, center_y)
        dist = float(np.hypot(dx, dy))
        center_score = max(0.0, 1.0 - dist)

        border_ratio = float(border_hits[label]) / max(1.0, float(area))
        border_penalty = max(0.02, 1.0 - min(0.95, border_ratio * 8.0))

        score = np.sqrt(float(area)) * (0.45 + 0.55 * center_score) * border_penalty
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1

    return labels == best_label


def _connected_to_core(core_mask: np.ndarray, allow_mask: np.ndarray) -> np.ndarray:
    """Keep only allowed-mask components that are connected to the core mask."""
    allow_u8 = allow_mask.astype(np.uint8)
    if not allow_u8.any():
        return core_mask

    labels_count, labels, _, _ = cv2.connectedComponentsWithStats(allow_u8, connectivity=8)
    if labels_count <= 1:
        return allow_mask

    seed_labels = np.unique(labels[core_mask & (labels > 0)])
    if seed_labels.size == 0:
        return core_mask

    return np.isin(labels, seed_labels)


def _build_border_background_mask(source_rgb: np.ndarray) -> np.ndarray:
    """
    Estimate background by learning border color and keeping only pixels
    connected to borders with similar color statistics.
    """
    h, w = source_rgb.shape[:2]
    band = max(8, int(min(h, w) * 0.04))

    lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    border_samples = np.concatenate(
        [
            lab[:band, :, :].reshape(-1, 3),
            lab[-band:, :, :].reshape(-1, 3),
            lab[:, :band, :].reshape(-1, 3),
            lab[:, -band:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    med = np.median(border_samples, axis=0)
    mad = np.median(np.abs(border_samples - med), axis=0) * 1.4826
    scale = np.maximum(mad, np.array([10.0, 6.0, 6.0], dtype=np.float32))

    dist2 = np.sum(((lab - med) / scale) ** 2, axis=2)
    candidate_bg = dist2 <= 8.5

    labels_count, labels, _, _ = cv2.connectedComponentsWithStats(candidate_bg.astype(np.uint8), connectivity=8)
    if labels_count <= 1:
        return candidate_bg

    border_labels = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    border_labels = np.unique(border_labels[border_labels > 0])
    if border_labels.size == 0:
        return np.zeros((h, w), dtype=bool)

    return np.isin(labels, border_labels)


def _refine_with_grabcut(
    source_rgb: np.ndarray,
    core_mask: np.ndarray,
    probable_fg: np.ndarray,
    probable_bg: np.ndarray,
) -> np.ndarray:
    """
    Use GrabCut with strong seeds:
    - core_mask: definite foreground
    - border ring: definite background
    - probable_bg/probable_fg as soft hints
    """
    h, w = source_rgb.shape[:2]
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[probable_bg] = cv2.GC_PR_BGD
    gc_mask[probable_fg] = cv2.GC_PR_FGD
    gc_mask[core_mask] = cv2.GC_FGD

    # Outer ring is always background in this dataset.
    ring = max(6, int(min(h, w) * 0.02))
    gc_mask[:ring, :] = cv2.GC_BGD
    gc_mask[-ring:, :] = cv2.GC_BGD
    gc_mask[:, :ring] = cv2.GC_BGD
    gc_mask[:, -ring:] = cv2.GC_BGD

    bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        return (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
    except cv2.error:
        return probable_fg


def get_alpha_bbox(image: Image.Image, threshold: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """
    Get tight bbox from alpha channel (non-transparent pixels).
    Returns (left, upper, right, lower) or None.
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image.split()[3], dtype=np.uint8)
    mask = _remove_tiny_components(alpha > threshold, min_area=max(32, int(alpha.size * 0.00003)))
    if mask is None:
        return None

    ys, xs = np.where(mask)
    left = int(xs.min())
    upper = int(ys.min())
    right = int(xs.max()) + 1
    lower = int(ys.max()) + 1
    return left, upper, right, lower


def _has_alpha_content(image: Image.Image, threshold: int = 1) -> bool:
    """Return True when the image contains visible alpha content."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    alpha = np.array(image.getchannel("A"), dtype=np.uint8)
    return bool(np.any(alpha > threshold))


def _keep_significant_components(
    binary_mask: np.ndarray,
    min_relative_area: float = 0.05,
    min_area: int = 128,
) -> Optional[np.ndarray]:
    """Keep the largest foreground component plus similarly large peers."""
    binary_u8 = binary_mask.astype(np.uint8)
    if not binary_u8.any():
        return None

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if labels_count <= 1:
        return binary_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    best_area = int(areas.max())
    keep_min_area = max(min_area, int(best_area * max(0.0, min_relative_area)))

    keep = np.zeros_like(binary_mask, dtype=bool)
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= keep_min_area:
            keep |= labels == label

    if not keep.any():
        keep = labels == (int(np.argmax(areas)) + 1)
    return keep


def _fill_enclosed_holes(
    mask: np.ndarray,
    max_hole_ratio: float = 0.03,
    small_hole_ratio: float = 0.004,
    min_rel_y: float = 0.18,
) -> np.ndarray:
    """
    Fill enclosed transparent holes inside a garment silhouette.
    Leaves large top-area openings (neck holes) untouched.
    """
    if not mask.any():
        return mask

    h, w = mask.shape
    inverse = (~mask).astype(np.uint8)
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(inverse, connectivity=8)

    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bbox_area = float(max(1, (right - left) * (bottom - top)))

    filled = mask.copy()
    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = stats[label]
        if x <= 0 or y <= 0 or (x + comp_w) >= w or (y + comp_h) >= h:
            # This inverse component connects to true outer background.
            continue

        area_ratio = float(area) / bbox_area
        _, cy = centroids[label]
        rel_y = (float(cy) - top) / max(1.0, float(bottom - top))

        if area_ratio <= small_hole_ratio or (area_ratio <= max_hole_ratio and rel_y >= min_rel_y):
            filled[labels == label] = True

    return filled


def _estimate_enclosed_hole_ratio(mask: np.ndarray) -> float:
    """Estimate ratio of enclosed holes inside the foreground bbox."""
    if not mask.any():
        return 0.0

    h, w = mask.shape
    inverse = (~mask).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)

    ys, xs = np.where(mask)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    bbox_area = float(max(1, (right - left) * (bottom - top)))

    enclosed_area = 0
    for label in range(1, labels_count):
        x, y, comp_w, comp_h, area = stats[label]
        if x <= 0 or y <= 0 or (x + comp_w) >= w or (y + comp_h) >= h:
            continue
        enclosed_area += int(area)

    return float(enclosed_area) / bbox_area


def _mask_needs_recovery(mask: np.ndarray, hole_ratio_trigger: float = 0.015) -> bool:
    """Detect fragmented/holed masks that should trigger a secondary model pass."""
    if not mask.any():
        return True

    labels_count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    hole_ratio = _estimate_enclosed_hole_ratio(mask)

    if labels_count > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max()) if areas.size else 0
        if largest > 0:
            large_parts = int(np.sum(areas >= max(128, int(largest * 0.05))))
            if large_parts >= 2:
                return True
            if labels_count >= 4 and hole_ratio >= 0.002:
                return True

    return hole_ratio >= max(0.0, hole_ratio_trigger)


def _merge_model_masks(primary_mask: np.ndarray, secondary_mask: np.ndarray, config: ProcessingConfig) -> np.ndarray:
    """Union primary + secondary masks, then keep substantial garment components."""
    merged = primary_mask | secondary_mask
    merged = _keep_significant_components(
        merged,
        min_relative_area=config.fragment_keep_ratio,
        min_area=128,
    )
    if merged is None:
        return primary_mask
    return merged


def _compose_rgba_with_mask(source_image: Image.Image, alpha: np.ndarray) -> Image.Image:
    """Create RGBA from source RGB and alpha mask; whiten fully transparent pixels."""
    source_rgba = np.array(source_image.convert("RGBA"), dtype=np.uint8)
    alpha_u8 = alpha.astype(np.uint8)
    source_rgba[:, :, 3] = alpha_u8

    transparent = alpha_u8 == 0
    source_rgba[transparent, 0] = 255
    source_rgba[transparent, 1] = 255
    source_rgba[transparent, 2] = 255
    return Image.fromarray(source_rgba, mode="RGBA")


def fallback_remove_background(source_image: Image.Image, config: ProcessingConfig) -> Image.Image:
    """
    Build a foreground mask directly from the source image when rembg returns
    an empty alpha result. This is tuned for catalog shots with a border-connected
    background and a centered garment.
    """
    rgba = np.array(source_image.convert("RGBA"), dtype=np.uint8)
    source_rgb = rgba[:, :, :3]

    border_bg = _build_border_background_mask(source_rgb)
    fg_mask = ~border_bg

    fg_u8 = (fg_mask.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg_mask = fg_u8 > 0

    main_mask = _best_component_mask(fg_mask)
    if main_mask is None:
        raise ValueError("Fallback foreground extraction could not detect the clothing item.")

    main_mask[0, :] = False
    main_mask[-1, :] = False
    main_mask[:, 0] = False
    main_mask[:, -1] = False

    alpha = np.where(main_mask, 255, 0).astype(np.uint8)
    rgba[:, :, 3] = alpha

    transparent = alpha == 0
    rgba[transparent, 0] = 255
    rgba[transparent, 1] = 255
    rgba[transparent, 2] = 255

    return Image.fromarray(rgba, mode="RGBA")


def recover_mask_with_secondary_model(
    source_image: Image.Image,
    primary_result: Image.Image,
    config: ProcessingConfig,
) -> Image.Image:
    """
    Optionally run a secondary rembg model and merge both masks when
    the primary mask appears fragmented or hole-heavy.
    """
    if not config.auto_recover_mask:
        return primary_result

    secondary_model = (config.secondary_rembg_model or "").strip()
    if not secondary_model or secondary_model == config.rembg_model:
        return primary_result

    primary_rgba = np.array(primary_result.convert("RGBA"), dtype=np.uint8)
    primary_alpha = primary_rgba[:, :, 3]
    primary_mask = primary_alpha > config.alpha_threshold
    if not _mask_needs_recovery(primary_mask, hole_ratio_trigger=config.hole_recovery_trigger_ratio):
        return primary_result

    try:
        logger.info(
            "Primary mask flagged for recovery; running secondary model '%s' for merge.",
            secondary_model,
        )
        secondary_config = replace(config, rembg_model=secondary_model, auto_recover_mask=False)
        secondary_result = remove_background(source_image, secondary_config)
        if config.smooth_edges:
            secondary_result = refine_alpha_edges(secondary_result, config.edge_smooth_radius)

        secondary_rgba = np.array(secondary_result.convert("RGBA"), dtype=np.uint8)
        secondary_alpha = secondary_rgba[:, :, 3]
        secondary_mask = secondary_alpha > config.alpha_threshold

        merged_mask = _merge_model_masks(primary_mask, secondary_mask, config)
        merged_mask = _fill_enclosed_holes(
            merged_mask,
            max_hole_ratio=config.hole_fill_max_ratio,
            small_hole_ratio=config.hole_fill_small_ratio,
            min_rel_y=config.hole_fill_min_rel_y,
        )

        merged_alpha = np.maximum(primary_alpha, secondary_alpha)
        merged_alpha = np.where(merged_mask, merged_alpha, 0).astype(np.uint8)
        return _compose_rgba_with_mask(source_image, merged_alpha)

    except Exception as exc:
        logger.warning("Secondary mask recovery failed; keeping primary mask. Error: %s", exc)
        return primary_result


def crop_to_alpha_content(
    image: Image.Image,
    threshold: int = 1,
    source_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Alpha-based tight crop:
    - detect non-transparent pixels from alpha
    - remove only tiny isolated specks
    - crop around the complete garment mask
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    rgba = np.array(image, dtype=np.uint8)
    alpha = rgba[:, :, 3]

    if not np.any(alpha > threshold):
        raise ValueError("No clothing detected after background removal (alpha is empty).")

    main_mask = _remove_tiny_components(alpha > threshold, min_area=max(32, int(alpha.size * 0.00003)))
    if main_mask is None:
        raise ValueError("No clothing detected after background removal (alpha is empty).")

    kernel = np.ones((3, 3), np.uint8)
    main_u8 = cv2.morphologyEx((main_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel, iterations=1)
    main_mask = main_u8 > 0

    # Remove accidental 1px border pixels.
    main_mask[0, :] = False
    main_mask[-1, :] = False
    main_mask[:, 0] = False
    main_mask[:, -1] = False

    new_alpha = np.where(main_mask, alpha, 0).astype(np.uint8)
    new_alpha[new_alpha < threshold] = 0
    cleaned = _compose_rgba_with_mask(image, new_alpha)

    bbox = get_alpha_bbox(cleaned, threshold=threshold)
    if bbox is None:
        raise ValueError("No visible clothing pixels available to crop.")
    return cleaned.crop(bbox)


def center_on_canvas(image: Image.Image, config: ProcessingConfig) -> Image.Image:
    """
    Resize to fit fixed canvas and center both axes with transparent background.
    No extra padding is applied.
    """
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid cropped image dimensions.")

    fit_scale = min(config.canvas_width / src_w, config.canvas_height / src_h)
    fit_scale *= max(0.01, min(1.0, config.fill_ratio))
    new_w = max(1, int(round(src_w * fit_scale)))
    new_h = max(1, int(round(src_h * fit_scale)))

    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (config.canvas_width, config.canvas_height), (0, 0, 0, 0))
    paste_x = (config.canvas_width - new_w) // 2
    paste_y = (config.canvas_height - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y), resized)
    return canvas


def crop_and_center(
    image: Image.Image,
    config: ProcessingConfig,
    source_image: Optional[Image.Image] = None,
) -> Image.Image:
    """Crop tightly by alpha and center in a transparent fixed canvas."""
    cropped = crop_to_alpha_content(
        image,
        threshold=config.alpha_threshold,
        source_image=source_image,
    )
    if config.use_cropped_size_canvas:
        return cropped
    return center_on_canvas(cropped, config)


def _load_heif_with_windows_decoder(input_path: str) -> Optional[Image.Image]:
    """
    Ask Windows Imaging Component to decode HEIC/HEIF using the system codec.
    This tends to select the primary image correctly when Explorer can preview it.
    """
    powershell_path = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell_path:
        return None

    script = """
param(
    [string]$InputPath,
    [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName PresentationCore

$resolvedInput = [System.IO.Path]::GetFullPath($InputPath)
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)

try {
    $frame = [Windows.Media.Imaging.BitmapFrame]::Create(
        [System.Uri]$resolvedInput,
        [Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
        [Windows.Media.Imaging.BitmapCacheOption]::OnLoad
    )
}
catch {
    $stream = [System.IO.File]::OpenRead($resolvedInput)
    try {
        $decoder = [Windows.Media.Imaging.BitmapDecoder]::Create(
            $stream,
            [Windows.Media.Imaging.BitmapCreateOptions]::PreservePixelFormat,
            [Windows.Media.Imaging.BitmapCacheOption]::OnLoad
        )
        if ($decoder.Frames.Count -lt 1) {
            throw 'No image frames found.'
        }

        $frame = $decoder.Frames |
            Sort-Object -Property @{ Expression = { $_.PixelWidth * $_.PixelHeight } } -Descending |
            Select-Object -First 1
    }
    finally {
        $stream.Dispose()
    }
}

$encoder = New-Object Windows.Media.Imaging.PngBitmapEncoder
$encoder.Frames.Add($frame)
$outStream = [System.IO.File]::Create($resolvedOutput)
try {
    $encoder.Save($outStream)
}
finally {
    $outStream.Dispose()
}
""".strip()

    temp_path = None
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        with tempfile.NamedTemporaryFile(suffix=".ps1", delete=False, mode="w", encoding="utf-8") as tmp_script:
            tmp_script.write(script)
            script_path = tmp_script.name

        subprocess.run(
            [
                powershell_path,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                script_path,
                input_path,
                temp_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,
        )

        with Image.open(temp_path) as opened:
            logger.info("Loaded HEIC/HEIF via Windows codec: %s", input_path)
            return ImageOps.exif_transpose(opened).convert("RGBA")
    except Exception as exc:
        logger.debug("Windows HEIC decode failed for %s: %s", input_path, exc)
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        if script_path and os.path.exists(script_path):
            try:
                os.remove(script_path)
            except OSError:
                pass


def _load_heif_with_ffmpeg(input_path: str) -> Optional[Image.Image]:
    """Decode HEIC/HEIF with ffmpeg when available."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    try:
        proc = subprocess.run(
            [
                ffmpeg_path,
                "-v",
                "error",
                "-i",
                input_path,
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,
        )
        if proc.stdout:
            with Image.open(io.BytesIO(proc.stdout)) as opened:
                logger.info("Loaded HEIC/HEIF via ffmpeg: %s", input_path)
                return ImageOps.exif_transpose(opened).convert("RGBA")
    except Exception as exc:
        logger.debug("FFmpeg HEIC decode failed for %s: %s", input_path, exc)

    return None


def load_input_image(input_path: str) -> Image.Image:
    """
    Load an input image and normalize orientation/mode.
    HEIC/HEIF first tries Pillow (with optional pillow-heif plugin) and then
    imageio as a fallback decoder when available.
    """
    suffix = Path(input_path).suffix.lower()
    try:
        with Image.open(input_path) as opened:
            return ImageOps.exif_transpose(opened).convert("RGBA")
    except (UnidentifiedImageError, OSError):
        if suffix not in HEIF_FORMATS:
            raise

        image = _load_heif_with_windows_decoder(input_path)
        if image is not None:
            return image

        image = _load_heif_with_ffmpeg(input_path)
        if image is not None:
            return image

        try:
            import imageio.v3 as iio

            frame = iio.imread(input_path)
            return ImageOps.exif_transpose(Image.fromarray(np.asarray(frame))).convert("RGBA")
        except Exception as fallback_exc:
            raise RuntimeError(
                "HEIC/HEIF input detected, but no decoder is available in this environment. "
                "Install pillow-heif (if available) or install FFmpeg and ensure `ffmpeg` is in PATH."
            ) from fallback_exc


def process_image(input_path: str, output_path: str, config: ProcessingConfig) -> bool:
    """
    Full pipeline:
    load -> remove bg -> refine edges -> tight alpha crop -> center -> save
    """
    image_name = Path(input_path).name
    total_start = time.perf_counter()

    try:
        logger.info("[%s] Step 1/5 load image", image_name)
        t0 = time.perf_counter()
        img = load_input_image(input_path)
        _save_debug_image(config, image_name, "original", img)
        logger.info("[%s] Step 1/5 done in %.2fs", image_name, time.perf_counter() - t0)

        logger.info("[%s] Step 2/5 remove background", image_name)
        t0 = time.perf_counter()
        img_nobg = remove_background(img, config)
        _save_debug_image(config, image_name, "raw_mask", _alpha_mask_image(np.array(img_nobg.getchannel("A"))))
        if not _has_alpha_content(img_nobg, threshold=config.alpha_threshold):
            logger.warning("[%s] rembg returned empty alpha; using source-image fallback mask", image_name)
            img_nobg = fallback_remove_background(img, config)
        logger.info("[%s] Step 2/5 done in %.2fs", image_name, time.perf_counter() - t0)

        if config.smooth_edges:
            logger.info("[%s] Step 3/5 refine alpha edges", image_name)
            t0 = time.perf_counter()
            img_nobg = refine_alpha_edges(img_nobg, config.edge_smooth_radius)
            logger.info("[%s] Step 3/5 done in %.2fs", image_name, time.perf_counter() - t0)

        if config.auto_recover_mask and (config.secondary_rembg_model or "").strip():
            logger.info("[%s] Step 3b/5 optional mask recovery", image_name)
            t0 = time.perf_counter()
            img_nobg = recover_mask_with_secondary_model(img, img_nobg, config)
            logger.info("[%s] Step 3b/5 done in %.2fs", image_name, time.perf_counter() - t0)

        logger.info("[%s] Step 3c/5 validate + refine garment mask", image_name)
        t0 = time.perf_counter()
        refined_alpha, validation = build_refined_alpha(img, img_nobg, config)
        if not np.any(refined_alpha > config.alpha_threshold):
            logger.warning("[%s] Refined alpha is empty; using source-image fallback mask", image_name)
            img_nobg = fallback_remove_background(img, config)
            refined_alpha, validation = build_refined_alpha(img, img_nobg, config)
        img_nobg = _compose_rgba_with_mask(img, refined_alpha)
        _save_debug_image(config, image_name, "refined_mask", _alpha_mask_image(refined_alpha))
        logger.info(
            "[%s] Step 3c/5 done in %.2fs (components=%s, solidity=%.3f, holes=%.3f%s)",
            image_name,
            time.perf_counter() - t0,
            validation.component_count,
            validation.bbox_solidity,
            validation.enclosed_hole_ratio,
            f", flagged={validation.reason}" if validation.is_suspicious else "",
        )

        logger.info("[%s] Step 4/5 alpha crop + center", image_name)
        t0 = time.perf_counter()
        try:
            img_final = crop_and_center(img_nobg, config, source_image=img)
        except ValueError as exc:
            if "alpha is empty" not in str(exc):
                raise
            logger.warning("[%s] Crop received empty alpha; retrying with source-image fallback mask", image_name)
            img_nobg = fallback_remove_background(img, config)
            if config.smooth_edges:
                img_nobg = refine_alpha_edges(img_nobg, config.edge_smooth_radius)
            refined_alpha, _ = build_refined_alpha(img, img_nobg, config)
            img_nobg = _compose_rgba_with_mask(img, refined_alpha)
            img_final = crop_and_center(img_nobg, config, source_image=img)
        _save_debug_image(config, image_name, "final", img_final)
        logger.info("[%s] Step 4/5 done in %.2fs", image_name, time.perf_counter() - t0)

        logger.info("[%s] Step 5/5 save output", image_name)
        t0 = time.perf_counter()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_format = config.output_format.upper()
        save_kwargs = {}
        if output_format == "WEBP":
            save_kwargs.update(
                quality=config.webp_quality,
                lossless=config.webp_lossless,
                method=6,
                exact=True,
            )
        img_final.save(output_path, format=output_format, **save_kwargs)
        logger.info("[%s] Step 5/5 done in %.2fs", image_name, time.perf_counter() - t0)
        logger.info("[%s] Finished in %.2fs", image_name, time.perf_counter() - total_start)
        return True

    except Exception as exc:
        logger.error("Failed to process %s: %s\n%s", input_path, exc, traceback.format_exc())
        return False


def collect_images(folder: str) -> list:
    """Return supported image paths in folder (non-recursive)."""
    folder = Path(folder)
    images = []
    for image_file in sorted(folder.iterdir()):
        if image_file.is_file() and image_file.suffix.lower() in SUPPORTED_FORMATS:
            images.append(str(image_file))
    return images


def build_output_path(input_path: str, output_folder: str, config: ProcessingConfig) -> str:
    """Build output path with the configured transparent image extension."""
    stem = Path(input_path).stem
    ext = ".png" if config.output_format.upper() == "PNG" else ".webp"
    return str(Path(output_folder) / (stem + ext))
