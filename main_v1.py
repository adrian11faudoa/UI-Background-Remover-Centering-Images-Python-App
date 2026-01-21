# Imports
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from rembg import remove

# Global Variables
original_img = None
processed_img = None

# Load Image Function
def load_image():
    global original_img
    path = filedialog.askopenfilename()
    if not path:
        return
    original_img = Image.open(path)
    show_preview(original_img, original=True)


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


# Save Image Function
def save_image():
    if processed_img is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        processed_img.save(path, "PNG")


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


# UI
root = tk.Tk()
root.title("Background Remover + Centering App")

# UI Buttons
tk.Button(root, text="Load Image", command=load_image).pack()
tk.Button(root, text="Remove Background", command=remove_background).pack()
tk.Button(root, text="Crop Empty Space", command=crop_space).pack()
tk.Button(root, text="Center Object", command=center_image).pack()
tk.Button(root, text="Save Image", command=save_image).pack()

# UI Image Previews
original_label = tk.Label(root)
original_label.pack(side="left")

result_label = tk.Label(root)
result_label.pack(side="right")

root.mainloop()
