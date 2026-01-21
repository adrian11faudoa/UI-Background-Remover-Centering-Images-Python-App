# Imports 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove

# Global Variables 
original_img = None
processed_img = None
image_paths = []
current_index = 0


# Load Multiple Images
def load_images():
    global image_paths, current_index
    paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
    if not paths:
        return
    image_paths = list(paths)
    current_index = 0
    load_image_at_index(current_index)

def load_image_at_index(i):
    global original_img, processed_img
    path = image_paths[i]
    original_img = Image.open(path).convert("RGBA")
    processed_img = None
    show_preview(original_img, original=True)
    result_label.config(image="")  # clear result preview


# Remove Background Function
def remove_background():
    global processed_img
    if original_img is None:
        return
    processed_img = remove(original_img)
    show_preview(processed_img, original=False)


# Crop Empty Space Function
def crop_space():
    global processed_img
    if processed_img is None:
        return
    bbox = processed_img.getbbox()
    processed_img = processed_img.crop(bbox)
    show_preview(processed_img, original=False)


# Center Image Function
def center_image():
    global processed_img
    if processed_img is None:
        return
    w, h = processed_img.size
    size = max(w, h)
    canvas = Image.new("RGBA", (size, size), (0,0,0,0))
    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(processed_img, (x, y), processed_img)
    processed_img = canvas
    show_preview(processed_img, original=False)


# All Processes Combined Function (BG Remove + Crop + Center Functions)
def all_processes():
    remove_background()
    crop_space()
    center_image()


# Save Image Function
def save_image():
    if processed_img is None:
        messagebox.showerror("Error", "No processed image available.")
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png")])
    if path:
        processed_img.save(path, "PNG")
        messagebox.showinfo("Saved", "Image saved successfully.")


# Show Preview Function
def show_preview(img, original=True):
    preview = img.copy()
    preview.thumbnail((250, 250))
    tk_img = ImageTk.PhotoImage(preview)
    if original:
        original_label.config(image=tk_img)
        original_label.image = tk_img
    else:
        result_label.config(image=tk_img)
        result_label.image = tk_img


# Next & Previous Image Functions
def next_image():
    global current_index
    if not image_paths:
        return
    if current_index < len(image_paths) - 1:
        current_index += 1
        load_image_at_index(current_index)

def prev_image():
    global current_index
    if not image_paths:
        return
    if current_index > 0:
        current_index -= 1
        load_image_at_index(current_index)


# UI
root = tk.Tk()
root.title("Background Remover + Centering App")

# UI Buttons
tk.Button(root, text="Select Images", command=load_images).pack(pady=4)
tk.Button(root, text="Remove Background", command=remove_background).pack(pady=4)
tk.Button(root, text="All Processes", command=all_processes).pack(pady=4)
tk.Button(root, text="Save Image", command=save_image).pack(pady=4)

# UI Navigation
nav_frame = tk.Frame(root)
nav_frame.pack(pady=5)
tk.Button(nav_frame, text="◀ Previous", command=prev_image).pack(side="left", padx=5)
tk.Button(nav_frame, text="Next ▶", command=next_image).pack(side="right", padx=5)

# UI Image Previews
preview_frame = tk.Frame(root)
preview_frame.pack()
original_label = tk.Label(preview_frame, text="Original")
original_label.pack(side="left", padx=10)
result_label = tk.Label(preview_frame, text="Result")
result_label.pack(side="right", padx=10)

root.mainloop()
