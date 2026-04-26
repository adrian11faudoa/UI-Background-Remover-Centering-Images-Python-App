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
from dataclasses import dataclass
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
            post_process_mask=True,
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
    mask = _best_component_mask(alpha > threshold)
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


def crop_to_alpha_content(
    image: Image.Image,
    threshold: int = 1,
    source_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Alpha-based tight crop:
    - detect non-transparent pixels from alpha
    - keep only largest connected component
    - crop with zero margins
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    rgba = np.array(image, dtype=np.uint8)
    alpha = rgba[:, :, 3]

    if not np.any(alpha > threshold):
        raise ValueError("No clothing detected after background removal (alpha is empty).")

    # Find a stable high-confidence core first, then grow only into connected
    # medium-alpha regions. This prevents border/background leaks.
    core_thresholds = [240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10]
    core_mask = None
    core_thr_used = max(2, threshold)
    for core_thr in core_thresholds:
        core_thr = max(core_thr, threshold)
        candidate = _best_component_mask(alpha >= core_thr)
        if candidate is None:
            continue
        area = int(candidate.sum())
        if area < max(80, int(alpha.size * 0.00005)):
            continue
        core_mask = candidate
        core_thr_used = core_thr
        break

    if core_mask is None:
        core_mask = _best_component_mask(alpha > threshold)
        if core_mask is None:
            raise ValueError("No clothing detected after background removal (alpha is empty).")

    soft_thr = max(threshold, min(24, max(8, core_thr_used // 3)))
    soft_mask = alpha >= soft_thr

    source_rgb = None
    border_bg = np.zeros_like(soft_mask, dtype=bool)
    if source_image is not None:
        source_rgb = np.array(source_image.convert("RGB"), dtype=np.uint8)
        if source_rgb.shape[:2] == soft_mask.shape:
            border_bg = _build_border_background_mask(source_rgb)
            # Background-like border areas are deprioritized, but we keep core.
            soft_mask = (soft_mask & (~border_bg)) | core_mask
        else:
            source_rgb = None

    main_mask = _connected_to_core(core_mask, soft_mask)
    if source_rgb is not None:
        main_mask = _refine_with_grabcut(
            source_rgb=source_rgb,
            core_mask=core_mask,
            probable_fg=main_mask,
            probable_bg=border_bg,
        )
        main_mask = _connected_to_core(core_mask, main_mask)

    main_mask = _best_component_mask(main_mask)
    if main_mask is None:
        main_mask = core_mask

    # Break tiny accidental bridges and smooth component edges.
    main_u8 = (main_mask.astype(np.uint8) * 255)
    main_u8 = cv2.morphologyEx(main_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    main_u8 = cv2.morphologyEx(main_u8, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    main_mask = main_u8 > 0
    main_mask = _connected_to_core(core_mask, main_mask)

    # Hard-clear 1px outer border noise that can create full-frame black margins.
    main_mask[0, :] = False
    main_mask[-1, :] = False
    main_mask[:, 0] = False
    main_mask[:, -1] = False

    # Keep only article pixels; hard-zero all background alpha.
    new_alpha = np.where(main_mask, alpha, 0).astype(np.uint8)
    new_alpha[new_alpha < threshold] = 0
    rgba[:, :, 3] = new_alpha

    # Avoid black halos in apps that preview transparent WEBP against black.
    transparent = new_alpha == 0
    rgba[transparent, 0] = 255
    rgba[transparent, 1] = 255
    rgba[transparent, 2] = 255

    cleaned = Image.fromarray(rgba, mode="RGBA")

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
        logger.info("[%s] Step 1/5 done in %.2fs", image_name, time.perf_counter() - t0)

        logger.info("[%s] Step 2/5 remove background", image_name)
        t0 = time.perf_counter()
        img_nobg = remove_background(img, config)
        if not _has_alpha_content(img_nobg, threshold=config.alpha_threshold):
            logger.warning("[%s] rembg returned empty alpha; using source-image fallback mask", image_name)
            img_nobg = fallback_remove_background(img, config)
        logger.info("[%s] Step 2/5 done in %.2fs", image_name, time.perf_counter() - t0)

        if config.smooth_edges:
            logger.info("[%s] Step 3/5 refine alpha edges", image_name)
            t0 = time.perf_counter()
            img_nobg = refine_alpha_edges(img_nobg, config.edge_smooth_radius)
            logger.info("[%s] Step 3/5 done in %.2fs", image_name, time.perf_counter() - t0)

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
            img_final = crop_and_center(img_nobg, config, source_image=img)
        logger.info("[%s] Step 4/5 done in %.2fs", image_name, time.perf_counter() - t0)

        logger.info("[%s] Step 5/5 save output", image_name)
        t0 = time.perf_counter()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        img_final.save(
            output_path,
            format="WEBP",
            quality=config.webp_quality,
            lossless=config.webp_lossless,
            method=6,
            exact=True,
        )
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
    """Build output path with .webp extension."""
    stem = Path(input_path).stem
    return str(Path(output_folder) / (stem + ".webp"))
