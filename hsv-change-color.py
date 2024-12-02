import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

class HSVColorChanger:
    def __init__(self, root):
        self.root = root
        self.root.title("Đổi màu áo")
        self.model = YOLO("yolo11n.pt")
        
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        self.original_label = ttk.Label(image_frame)
        self.original_label.grid(row=0, column=0, padx=5)
        ttk.Label(image_frame, text="Camera").grid(row=1, column=0)
        
        self.processed_label = ttk.Label(image_frame)
        self.processed_label.grid(row=0, column=1, padx=5)
        ttk.Label(image_frame, text="Ảnh đã xử lý").grid(row=1, column=1)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        self.camera_btn = ttk.Button(control_frame, text="Bật camera", command=self.toggle_camera)
        self.camera_btn.grid(row=0, column=0, padx=5)
        
        ttk.Label(control_frame, text="Chọn màu:").grid(row=0, column=1, padx=5)
        self.color_combo = ttk.Combobox(control_frame, values=['black', 'white', 'red', 'blue', 'green'])
        self.color_combo.grid(row=0, column=2, padx=5)
        self.color_combo.set('black')  # Giá trị mặc định
        self.color_combo.bind('<<ComboboxSelected>>', lambda e: self.create_mask())
        
        ttk.Label(control_frame, text="Hue:").grid(row=1, column=0, padx=5)
        self.hue_scale = ttk.Scale(control_frame, from_=0, to=179, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.hue_scale.grid(row=1, column=1, padx=5)
        
        ttk.Label(control_frame, text="Saturation:").grid(row=2, column=0, padx=5)
        self.sat_scale = ttk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.sat_scale.grid(row=2, column=1, padx=5)
        
        ttk.Label(control_frame, text="Value:").grid(row=3, column=0, padx=5)
        self.val_scale = ttk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.val_scale.grid(row=3, column=1, padx=5)
        
        self.camera = None
        self.is_camera_on = False
        self.image = None
        self.mask = None
        
        self.update()
        
    def toggle_camera(self):
        if self.is_camera_on:
            if self.camera is not None:
                self.camera.release()
            self.camera = None
            self.is_camera_on = False
            self.camera_btn.configure(text="Bật camera")
        else:
            self.camera = cv2.VideoCapture(0)
            self.is_camera_on = True
            self.camera_btn.configure(text="Tắt camera")
            
    def update(self):
        if self.is_camera_on and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                results = self.model.track(frame, persist=True, classes=0)
                self.image = frame
                self.display_image(self.image, self.original_label)
                self.create_mask()
        
        self.root.after(10, self.update)
            
    def create_mask(self):
        if self.image is None:
            return
            
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        color_ranges = {
            'black': (np.array([0, 0, 0]), np.array([180, 255, 45])),
            'white': (np.array([0, 0, 180]), np.array([180, 50, 255])), 
            'red': (np.array([0, 70, 50]), np.array([20, 255, 255])),
            'blue': (np.array([85, 50, 50]), np.array([145, 255, 255])),
            'green': (np.array([35, 70, 50]), np.array([85, 255, 255]))
        }
        
        selected_color = self.color_combo.get()
        lower, upper = color_ranges[selected_color]
        
        self.mask = cv2.inRange(hsv, lower, upper)
        
        kernel = np.ones((5,5), np.uint8)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        self.mask = cv2.morphologyEx(self.mask, cv2.MORPH_OPEN, kernel)
        
        self.update_color(None)
        
    def update_color(self, event):
        if self.image is None or self.mask is None:
            return
            
        new_hue = int(self.hue_scale.get())
        new_sat = int(self.sat_scale.get())
        new_val = int(self.val_scale.get())
        
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        hsv[self.mask > 0, 0] = new_hue
        hsv[self.mask > 0, 1] = new_sat
        hsv[self.mask > 0, 2] = new_val
        
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
        self.display_image(result, self.processed_label)
        
    def display_image(self, cv_img, label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(rgb_img)
        
        width, height = pil_img.size
        max_size = 400
        if width > max_size or height > max_size:
            ratio = min(max_size/width, max_size/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            pil_img = pil_img.resize((new_width, new_height))
        
        photo = ImageTk.PhotoImage(pil_img)
        
        label.configure(image=photo)
        label.image = photo

    def __del__(self):
        if self.camera is not None:
            self.camera.release()
if __name__ == "__main__":
    root = tk.Tk()
    app = HSVColorChanger(root)
    root.mainloop()
