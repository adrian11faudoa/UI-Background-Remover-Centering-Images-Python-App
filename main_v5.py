# Imports 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove
import threading
import os
from pillow_heif import register_heif_opener
register_heif_opener()

# Global Variables 
image_paths = []
original_imgs = []
processed_imgs = []
current_index = 0
preview_size = 350  # larger previews


# Load Multiple Images
def load_images():
    global image_paths, original_imgs, processed_imgs, current_index
    paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.heic *.heif")])
    if not paths:
        return
    image_paths = list(paths)
    # preload originals (converted to RGBA) and clear processed list
    original_imgs = [Image.open(p).convert("RGBA") for p in image_paths]
    processed_imgs = [None] * len(original_imgs)
    current_index = 0
    load_image_at_index(current_index)

def load_image_at_index(i):
    global current_index
    if not (0 <= i < len(original_imgs)):
        return
    current_index = i
    show_original_preview(original_imgs[i])
    # show processed preview if exists, otherwise clear
    if processed_imgs[i] is not None:
        show_result_preview(processed_imgs[i])
    else:
        clear_result_preview()
    update_status()


# Update Status Label
def update_status():
    if image_paths:
        status_var.set(f"{current_index+1} / {len(image_paths)} images")
    else:
        status_var.set("No images loaded")


# Remove Background Single Function 
def remove_background_single(img):
    return remove(img).convert("RGBA")


# Crop Empty Space Single Function 
def crop_space_single(img):
    bbox = img.getbbox()
    if not bbox:
        return img
    return img.crop(bbox)


# Center Image Single Function 
def center_image_single(img):
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(img, (x, y), img)
    return canvas


# Remove Background Function
def remove_background():
    global processed_imgs
    if not original_imgs:
        return
    img = original_imgs[current_index]
    processed_imgs[current_index] = remove_background_single(img)
    show_result_preview(processed_imgs[current_index])


# Crop Empty Space Function 
def crop_space():
    global processed_imgs
    if processed_imgs[current_index] is None:
        return
    processed_imgs[current_index] = crop_space_single(processed_imgs[current_index])
    show_result_preview(processed_imgs[current_index])


# Center Image Function
def center_image():
    global processed_imgs
    if processed_imgs[current_index] is None:
        return
    processed_imgs[current_index] = center_image_single(processed_imgs[current_index])
    show_result_preview(processed_imgs[current_index])


# All Processes Combined Function (BG Remove + Crop + Center Functions)
def all_processes():
    remove_background()
    crop_space()
    center_image()


# Batch Processing (THREADING)
def batch_process_all():
    if not original_imgs:
        messagebox.showerror("Error", "No images loaded.")
        return
    out_folder = filedialog.askdirectory(title="Select output folder")
    if not out_folder:
        return
    # Disable UI buttons during processing
    for widget in button_list:
        widget.config(state="disabled")
    progress_label.config(text="Starting batch...")

    def worker():
        total = len(original_imgs)
        for i in range(total):
            try:
                img = original_imgs[i]
                result = remove_background_single(img)
                result = crop_space_single(result)
                result = center_image_single(result)
                processed_imgs[i] = result
                # Save file
                # Always save as PNG (preserves transparency)
                base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
                save_path = os.path.join(out_folder, f"{base_name}.png")
                result.save(save_path, "PNG")
            except Exception as e:
                print(f"Error at image {i}: {e}")
            progress = (i + 1) / total * 100
            progress_var.set(progress)
            progress_label.config(text=f"Processing {i+1} / {total} images...")
        progress_label.config(text="Done!")
        messagebox.showinfo("Batch Complete", "All images processed and saved.")
        # Re-enable buttons
        for widget in button_list:
            widget.config(state="normal")
    threading.Thread(target=worker, daemon=True).start()


# Save Image (for current processed image)
def save_image():
    if processed_imgs[current_index] is None:
        messagebox.showerror("Error", "No processed image to save.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        processed_imgs[current_index].save(path, "PNG")
        messagebox.showinfo("Saved", "Image saved successfully.")


# Show Preview Function + ZOOM
def show_thumbnail(img):
    preview = img.copy()
    preview.thumbnail((preview_size, preview_size))
    return ImageTk.PhotoImage(preview)

def show_original_preview(img):
    tk_img = show_thumbnail(img)
    original_label.config(image=tk_img)
    original_label.image = tk_img

def show_result_preview(img):
    tk_img = show_thumbnail(img)
    result_label.config(image=tk_img)
    result_label.image = tk_img

def clear_result_preview():
    result_label.config(image="", text="Result")
    result_label.image = None


# Navigation Functions
def next_image():
    if current_index < len(original_imgs) - 1:
        load_image_at_index(current_index + 1)

def prev_image():
    if current_index > 0:
        load_image_at_index(current_index - 1)


# UI 
root = tk.Tk()
root.title("Background Remover + Auto-Center + Batch App")

# UI Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=6)
btn_load = tk.Button(btn_frame, text="Select Images", command=load_images)
btn_bg = tk.Button(btn_frame, text="Remove Background", command=remove_background)
btn_auto = tk.Button(btn_frame, text="All Processes", command=all_processes)
btn_save = tk.Button(btn_frame, text="Save Image", command=save_image)
btn_batch = tk.Button(btn_frame, text="Batch Process All", command=batch_process_all)
btn_load.grid(row=0, column=0, padx=4)
btn_bg.grid(row=0, column=1, padx=4)
btn_auto.grid(row=0, column=2, padx=4)
btn_save.grid(row=0, column=3, padx=4)
btn_batch.grid(row=0, column=4, padx=4)
button_list = [btn_load, btn_bg, btn_auto, btn_save, btn_batch]

# UI Navigation
nav_frame = tk.Frame(root)
nav_frame.pack(pady=5)
tk.Button(nav_frame, text="◀ Previous", command=prev_image).pack(side="left", padx=6)
tk.Button(nav_frame, text="Next ▶", command=next_image).pack(side="right", padx=6)

# UI Status
status_var = tk.StringVar(value="No images loaded")
status_label = tk.Label(root, textvariable=status_var)
status_label.pack()

# UI Progress bar
progress_var = tk.DoubleVar(value=0)
progress_bar = tk.Frame(root, height=20, width=300, bd=1, relief="sunken")
progress_bar.pack(pady=5)
progress_label = tk.Label(root, text="")
progress_label.pack()
def update_progress_bar(*args):
    pct = progress_var.get()
    fill_width = int((pct / 100) * 300)
    for widget in progress_bar.winfo_children():
        widget.destroy()
    fill = tk.Frame(progress_bar, bg="green", width=fill_width, height=20)
    fill.pack(side="left")
progress_var.trace_add("write", update_progress_bar)

# UI Image Previews
preview_frame = tk.Frame(root)
preview_frame.pack()
original_label = tk.Label(preview_frame, text="Original", bd=1, relief="solid")
original_label.pack(side="left", padx=10)
result_label = tk.Label(preview_frame, text="Result", bd=1, relief="solid")
result_label.pack(side="right", padx=10)

root.mainloop()
