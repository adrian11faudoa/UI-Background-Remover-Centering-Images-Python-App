# Imports 
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rembg import remove

# Global Variables 
original_imgs = []    # list of PIL.Image originals (RGBA)
processed_imgs = []   # list of PIL.Image processed (RGBA) or None
image_paths = []
current_index = 0


# Load Multiple Images
def load_images():
    global image_paths, original_imgs, processed_imgs, current_index
    paths = filedialog.askopenfilenames(
        title="Select images",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
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
    show_original_preview(original_imgs[current_index])
    # show processed preview if exists, otherwise clear
    if processed_imgs[current_index] is not None:
        show_result_preview(processed_imgs[current_index])
    else:
        clear_result_preview()
    update_status()


# Update Status Label
def update_status():
    if image_paths:
        status_var.set(f"{current_index + 1} / {len(image_paths)}")
    else:
        status_var.set("No images loaded")


# Remove Background Function (for current image)
def remove_background():
    if not original_imgs:
        return
    img = original_imgs[current_index]
    try:
        result = remove(img)
    except Exception as e:
        messagebox.showerror("Error", f"Background removal failed:\n{e}")
        return
    processed_imgs[current_index] = result.convert("RGBA")
    show_result_preview(processed_imgs[current_index])


# Crop Empty Space Function (for current processed image)
def crop_space():
    if not processed_imgs or processed_imgs[current_index] is None:
        messagebox.showerror("Error", "No processed image to crop. Run Remove Background first.")
        return
    img = processed_imgs[current_index]
    bbox = img.getbbox()
    if not bbox:
        messagebox.showinfo("Info", "Image appears empty after background removal.")
        return
    cropped = img.crop(bbox)
    processed_imgs[current_index] = cropped
    show_result_preview(cropped)


# Center Image Function (for current processed image)
def center_image():
    if not processed_imgs or processed_imgs[current_index] is None:
        messagebox.showerror("Error", "No processed image to center. Run Remove Background first.")
        return
    img = processed_imgs[current_index]
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(img, (x, y), img)
    processed_imgs[current_index] = canvas
    show_result_preview(canvas)


# All Processes Combined Function (BG Remove + Crop + Center Functions)
def all_processes():
    remove_background()
    # after removal, crop and center only if removal succeeded
    if processed_imgs[current_index] is not None:
        crop_space()
        center_image()


# Save Image (for current processed image)
def save_image():
    if not processed_imgs or processed_imgs[current_index] is None:
        messagebox.showerror("Error", "No processed image available to save.")
        return
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG", "*.png")])
    if path:
        processed_imgs[current_index].save(path, "PNG")
        messagebox.showinfo("Saved", "Image saved successfully.")


# Show Preview Function
def show_thumbnail(img, max_size=(250, 250)):
    preview = img.copy()
    preview.thumbnail(max_size)
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


# Next & Previous Image Functions
def next_image():
    if not image_paths:
        return
    if current_index < len(image_paths) - 1:
        load_image_at_index(current_index + 1)

def prev_image():
    if not image_paths:
        return
    if current_index > 0:
        load_image_at_index(current_index - 1)


# PREV / NEXT BY SAVING CHANGES (optional helper)
# (not necessary since we write processed_imgs immediately after operations)


# UI 
root = tk.Tk()
root.title("Background Remover + Centering App")

# UI Buttons
btn_frame = tk.Frame(root)
btn_frame.pack(pady=6)
tk.Button(btn_frame, text="Select Images", command=load_images).grid(row=0, column=0, padx=4)
tk.Button(btn_frame, text="Remove Background", command=remove_background).grid(row=0, column=1, padx=4)
tk.Button(btn_frame, text="All Processes", command=all_processes).grid(row=0, column=2, padx=4)
tk.Button(btn_frame, text="Save Image", command=save_image).grid(row=0, column=3, padx=4)

# UI Navigation
nav_frame = tk.Frame(root)
nav_frame.pack(pady=5)
tk.Button(nav_frame, text="◀ Previous", command=prev_image).pack(side="left", padx=6)
tk.Button(nav_frame, text="Next ▶", command=next_image).pack(side="right", padx=6)

#UI Status 
status_var = tk.StringVar(value="No images loaded")
status_label = tk.Label(root, textvariable=status_var)
status_label.pack()

# UI Image Previews
preview_frame = tk.Frame(root)
preview_frame.pack(pady=8)
original_label = tk.Label(preview_frame, text="Original", width=35, height=15, bd=1, relief="solid")
original_label.pack(side="left", padx=8)
result_label = tk.Label(preview_frame, text="Result", width=35, height=15, bd=1, relief="solid")
result_label.pack(side="right", padx=8)

root.mainloop()
