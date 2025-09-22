# draw_digit_pixel_data.py
# Draw a digit, see 28x28 pixel data, copy/save CSV, save 28x28 PNG

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import csv

# ---------- Config ----------
CANVAS_SIZE = 280
GRID_SIZE = 28
CELL = CANVAS_SIZE // GRID_SIZE
PEN_WIDTH = 18

# ---------- Preprocessing ----------
def preprocess_pil_image(pil_img):
    img = pil_img.convert("L")
    img_inv = ImageOps.invert(img)
    bbox = img_inv.getbbox()
    if bbox is None:
        return np.zeros((28,28), dtype=np.float32)
    cropped = img_inv.crop(bbox)
    max_side = max(cropped.size)
    scale = 20.0 / max_side
    new_w = max(1, int(round(cropped.size[0]*scale)))
    new_h = max(1, int(round(cropped.size[1]*scale)))
    resized = cropped.resize((new_w,new_h), Image.LANCZOS)
    new_img = Image.new("L", (28,28), 0)
    upper_left = ((28 - resized.width)//2, (28 - resized.height)//2)
    new_img.paste(resized, upper_left)
    arr = np.array(new_img).astype(np.float32)/255.0
    return arr

# ---------- GUI ----------
class DrawApp:
    def __init__(self, root):
        self.root = root
        root.title("Draw Digit — Get Pixel Data")

        main = tk.Frame(root)
        main.pack(padx=6, pady=6)

        # Canvas
        self.canvas = tk.Canvas(main, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="white")
        self.canvas.grid(row=0, column=0, rowspan=3)
        for i in range(0, CANVAS_SIZE, CELL):
            self.canvas.create_line([(i,0),(i,CANVAS_SIZE)], fill="#e6e6e6")
            self.canvas.create_line([(0,i),(CANVAS_SIZE,i)], fill="#e6e6e6")

        # Right side: grid preview and buttons
        right = tk.Frame(main)
        right.grid(row=0, column=1, sticky="n")

        self.grid_canvas = tk.Canvas(right,width=CANVAS_SIZE,height=CANVAS_SIZE,bg="white")
        self.grid_canvas.pack()

        btns = tk.Frame(right)
        btns.pack(pady=4)
        tk.Button(btns,text="Clear",command=self.clear).grid(row=0,column=0,padx=3)

        # Pixel data
        data_frame = tk.Frame(main)
        data_frame.grid(row=0,column=2,sticky="n")
        tk.Label(data_frame,text="Pixel Data (28x28)").pack()
        self.data_text = tk.Text(data_frame,width=28*3,height=18,font=("Courier",9))
        self.data_text.pack()
        export_btns = tk.Frame(data_frame)
        export_btns.pack(pady=4)
        tk.Button(export_btns,text="Copy",command=self.copy_pixels).grid(row=0,column=0,padx=3)
        tk.Button(export_btns,text="Save CSV",command=self.save_csv).grid(row=0,column=1,padx=3)
        tk.Button(export_btns,text="Save Image",command=self.save_image).grid(row=0,column=2,padx=3)

        # PIL image for drawing
        self.image = Image.new("L",(CANVAS_SIZE,CANVAS_SIZE),255)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x = None
        self.last_y = None

        self.canvas.bind("<B1-Motion>",self.paint)
        self.canvas.bind("<Button-1>",self.set_last)
        self.canvas.bind("<ButtonRelease-1>",self.reset_last)

        self.update_pixel_grid()

    def set_last(self,e):
        self.last_x,self.last_y = e.x,e.y
    def reset_last(self,e):
        self.last_x,self.last_y = None,None
    def paint(self,e):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x,self.last_y,e.x,e.y,width=PEN_WIDTH,fill="black",capstyle=tk.ROUND)
            self.draw.line([self.last_x,self.last_y,e.x,e.y],fill=0,width=PEN_WIDTH)
        self.last_x,self.last_y = e.x,e.y
        self.update_pixel_grid()

    def update_pixel_grid(self):
        inv = ImageOps.invert(self.image)
        small = inv.resize((GRID_SIZE,GRID_SIZE),Image.LANCZOS)
        arr = np.array(small).astype(np.float32)/255.0
        self.grid_canvas.delete("all")
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                v = arr[r,c]
                gray = int(round(255*(1-v)))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0,y0=c*CELL,r*CELL
                self.grid_canvas.create_rectangle(x0,y0,x0+CELL,y0+CELL,fill=color,outline="")
        self.data_text.delete("1.0",tk.END)
        for r in range(GRID_SIZE):
            row = ", ".join(f"{v:.3f}" for v in arr[r])
            self.data_text.insert(tk.END,row+"\n")

    def clear(self):
        self.canvas.delete("all")
        for i in range(0,CANVAS_SIZE,CELL):
            self.canvas.create_line([(i,0),(i,CANVAS_SIZE)],fill="#e6e6e6")
            self.canvas.create_line([(0,i),(CANVAS_SIZE,i)],fill="#e6e6e6")
        self.draw.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE],fill=255)
        self.update_pixel_grid()

    def copy_pixels(self):
        text = self.data_text.get("1.0",tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied","Pixel data copied.")

    def save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv")
        if not path: return
        text = self.data_text.get("1.0",tk.END).strip().splitlines()
        with open(path,"w",newline="") as f:
            writer = csv.writer(f)
            for line in text:
                writer.writerow(line.split(", "))
        messagebox.showinfo("Saved",f"CSV saved to {path}")

    def save_image(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG files","*.png")])
        if not path: return
        inv = ImageOps.invert(self.image)
        small = inv.resize((28,28),Image.LANCZOS)
        small.save(path)
        messagebox.showinfo("Saved",f"28x28 image saved to {path}")

if __name__=="__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()
