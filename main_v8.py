# Imports 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove, new_session
import threading
import os
import numpy as np
from pillow_heif import register_heif_opener
register_heif_opener()
bg_session = new_session("isnet-general-use")


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
    original_imgs = [None] * len(image_paths)
    processed_imgs = [None] * len(image_paths)
    current_index = 0
    load_image_at_index(current_index)

def get_original_image(i):
    if original_imgs[i] is None:
        original_imgs[i] = Image.open(image_paths[i]).convert("RGBA")
    return original_imgs[i]

def load_image_at_index(i):
    global current_index
    if not (0 <= i < len(original_imgs)):
        return
    current_index = i
    img = get_original_image(i)
    show_original_preview(img)
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
    return remove(img, session=bg_session).convert("RGBA")


# Center Image Single Function 
def center_image_single(img):
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(img, (x, y), img)
    return canvas


# Crop Empty Space Single Function 
def crop_space_single(img, alpha_threshold=10):
    """
    Crop transparent borders using alpha channel threshold.
    This avoids soft-edge / shadow centering issues.
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    alpha = img.split()[-1]
    bbox = alpha.point(lambda p: 255 if p > alpha_threshold else 0).getbbox()
    if not bbox:
        return img
    return img.crop(bbox)


# Remove Background Function
def remove_background():
    if not image_paths:
        return
    btn_bg.config(state="disabled")

    def worker():
        img = get_original_image(current_index)
        result = remove_background_single(img)
        result = center_image_single(result)
        result = crop_space_single(result)

        processed_imgs[current_index] = result
        # Update UI safely from main thread
        root.after(0, lambda: finish(result))

    def finish(result):
        show_result_preview(result)
        btn_bg.config(state="normal")
    threading.Thread(target=worker, daemon=True).start()


# Remove Background All Images Function
def process_all_images():
    if not original_imgs:
        messagebox.showerror("Error", "No images loaded.")
        return
    for widget in button_list:
        widget.config(state="disabled")
    progress_label.config(text="Processing all images...")

    def worker():
        total = len(original_imgs)
        for i in range(total):
            try:
                img = get_original_image(i)
                result = remove_background_single(img)
                result = center_image_single(result)
                result = crop_space_single(result)
                processed_imgs[i] = result
            except Exception as e:
                print(f"Error processing image {i}: {e}")
            progress_var.set((i + 1) / total * 100)
            progress_label.config(
                text=f"Processing {i+1} / {total} images..."
            )
        root.after(0, finish)
    def finish():
        progress_label.config(text="Processing complete!")
        messagebox.showinfo("Done", "All images processed.")
        for widget in button_list:
            widget.config(state="normal")
    threading.Thread(target=worker, daemon=True).start()


# Save Image Function (for current processed image)
def save_image():
    if processed_imgs[current_index] is None:
        messagebox.showerror("Error", "No processed image to save.")
        return
    default_name = os.path.splitext(
        os.path.basename(image_paths[current_index]))[0] + ".png"
    path = filedialog.asksaveasfilename(
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[("PNG Image", "*.png")])
    if not path:
        return
    try:
        processed_imgs[current_index].save(path, "PNG")
        messagebox.showinfo("Saved", f"Image saved:\n{path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save image:\n{e}")


#Save All Images Function
def save_all_images():
    if not original_imgs:
        messagebox.showerror("Error", "No images loaded.")
        return
    out_folder = filedialog.askdirectory(title="Select output folder")
    if not out_folder:
        return
    for widget in button_list:
        widget.config(state="disabled")
    progress_label.config(text="Saving images...")
    def worker():
        total = len(original_imgs)
        for i in range(total):
            try:
                img_to_save = (
                    processed_imgs[i]
                    if processed_imgs[i] is not None
                    else get_original_image(i))
                base_name = os.path.splitext(
                    os.path.basename(image_paths[i]))[0]
                save_path = os.path.join(out_folder, f"{base_name}.png")
                img_to_save.save(save_path, "PNG")
            except Exception as e:
                print(f"Error saving image {i}: {e}")
            progress_var.set((i + 1) / total * 100)
            progress_label.config(
                text=f"Saving {i+1} / {total} images..."
            )
        root.after(0, finish)
    def finish():
        progress_label.config(text="Save complete!")
        messagebox.showinfo("Done", "All images saved.")
        for widget in button_list:
            widget.config(state="normal")
    threading.Thread(target=worker, daemon=True).start()

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


# UI Setup
root = tk.Tk()
root.title("Background Remover App")

# UI Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=6)
btn_load = tk.Button(btn_frame, text="Select Images", command=load_images)
btn_bg = tk.Button(btn_frame, text="Remove Background", command=remove_background)
btn_save = tk.Button(btn_frame, text="Save Image", command=save_image)
btn_batch = tk.Button(btn_frame, text="Remove Background All", command=process_all_images)
btn_save_all = tk.Button(btn_frame, text="Save All Images", command=save_all_images)

btn_load.grid(row=0, column=0, padx=4)
btn_bg.grid(row=0, column=1, padx=4)
btn_save.grid(row=0, column=2, padx=4)
btn_batch.grid(row=0, column=3, padx=4)
btn_save_all.grid(row=0, column=4, padx=4)

button_list = [btn_load, btn_bg, btn_save, btn_batch, btn_save_all]

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
