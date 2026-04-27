# Clothes Photo Editor — Background Removal, Centering, Auto-crop, WEBP convert

A desktop batch photo editor built specifically for **e-commerce clothing product images**.

Removes backgrounds, centers items, and normalises the entire catalogue to a consistent canvas — producing PNG/WEBP files with **true alpha-channel transparency**.

## Features

| Feature | Details |
|---|---|
| Background removal    | `rembg` + u2net model, alpha-matting for fabric/hair edges    |
| Edge refinement       | Morphological cleanup + Gaussian anti-aliasing                |
| Auto-crop             | Tight bounding-box crop, no empty margins                     |
| Centering             | Item perfectly centred on fixed canvas                        |
| Consistent scaling    | Clothing occupies configurable % of canvas height             |
| Batch processing      | Full folder, with progress bar and safe cancel                |
| Output                | PNG (lossless) or WEBP (90 quality), RGBA transparency        |


## Installation

### 1. Python ≥ 3.9 required
python --version

### 2. Install dependencies
pip install -r requirements.txt

> **GPU acceleration (optional but recommended for speed)**
> `rembg[gpu]` uses ONNX Runtime with CUDA. If you don't have a GPU, install `rembg` instead:
> pip install rembg

### 3. First run — model download
On first use, `rembg` will automatically download the **u2net model** (~170 MB). This happens once and is cached in `~/.u2net/`.


## Running the App
python main.py

.\.venv\Scripts\activate

deactivate


## Usage

1. **Input folder** — select the folder containing your raw clothing photos  
2. **Output folder** — where processed images will be saved  
3. **Format** — PNG (lossless transparency) or WEBP (smaller files)  
4. **Canvas** — default 1200×1600 px (portrait, standard e-commerce)  
5. **Fill %** — how much of the canvas height the item should occupy (default 85%)  
6. Click **▶ Start Processing**

Progress is shown as `done / total` with a percentage bar. Click **✕ Cancel** to stop safely after the current image finishes.

---

## Output

All output images are:
- **RGBA** (true alpha transparency — no white background)
- Centred on a fixed canvas
- Consistently scaled across the catalogue
- Saved in the output folder with the same filename + `.png` or `.webp`

---

## Supported Input Formats

`.jpg` `.jpeg` `.png` `.webp` `.bmp` `.tiff` `.tif` `.heic` `.heif`

`HEIC/HEIF` decoding first tries the Windows system codec, then Pillow plugins when available (`pillow-heif`), and finally `ffmpeg` if it is installed in your system `PATH`.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: rembg` | Run `pip install rembg` |
| Slow processing | Install GPU version: `pip install rembg[gpu]` |
| White halo on edges | Try images with better contrast between clothing and background |
| Item too small/large | Adjust the **Fill %** slider |
| App won't start | Make sure `tkinter` is installed: `python -m tkinter` |

---

## Project Structure

```
clothing_editor/
├── main.py                  # Entry point
├── requirements.txt
├── README.md
├── processing/
│   ├── pipeline.py          # Image processing (bg removal, crop, center)
│   └── batch.py             # Background thread batch runner
└── ui/
    └── app_window.py        # Tkinter UI
```
