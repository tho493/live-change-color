import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ultralyticsplus import YOLO

class HSVColorChanger:
    def __init__(self, root):
        self.root = root
        self.root.title("Đổi màu áo")
        self.model = YOLO("kesimeg/yolov8n-clothing-detection")
        
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
        
        ttk.Label(control_frame, text="Chọn phân loại:").grid(row=1, column=0, padx=5)
        self.classification_combo = ttk.Combobox(control_frame, values=['accessories', 'bags', 'clothing', 'shoes'])
        self.classification_combo.grid(row=1, column=1, padx=5)
        self.classification_combo.set('clothing')  # Giá trị mặc định
        self.classification_combo.bind('<<ComboboxSelected>>', lambda e: self.create_mask())
        
        ttk.Label(control_frame, text="Hue:").grid(row=3, column=0, padx=5)
        self.hue_scale = ttk.Scale(control_frame, from_=0, to=179, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.hue_scale.grid(row=3, column=1, padx=5)
        
        ttk.Label(control_frame, text="Saturation:").grid(row=4, column=0, padx=5)
        self.sat_scale = ttk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.sat_scale.grid(row=4, column=1, padx=5)
        
        ttk.Label(control_frame, text="Value:").grid(row=5, column=0, padx=5)
        self.val_scale = ttk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, length=200,
                                 command=self.update_color)
        self.val_scale.grid(row=5, column=1, padx=5)
        
        self.color_label = ttk.Label(control_frame, text="Màu trung bình:")
        self.color_label.grid(row=6, column=0, columnspan=2, padx=5)
        
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
                self.image = cv2.flip(frame, 1)
                self.display_image(self.image, self.original_label)
                self.create_mask()
        
        self.root.after(10, self.update)
    def create_mask(self):
        if self.image is None:
            return

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        class_ranges = {'accessories': 0, 'bags': 1, 'clothing': 2, 'shoes': 3}

        selected_class = class_ranges[self.classification_combo.get()]  
        self.results_yolo = self.model.track(self.image, classes=selected_class)

        self.mask = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
        
        for box in self.results_yolo[0].boxes:
            if box.conf[0] > 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Tìm màu trung bình của tâm vùng box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                avg_color = hsv[center_y + int(center_y * 0.1), center_x]
                # Đảm bảo rằng các giá trị màu nằm trong phạm vi HSV
                avg_color = np.maximum(np.minimum(avg_color, 255), 0)
                rgb_color = cv2.cvtColor(np.uint8([[[avg_color[0], avg_color[1], avg_color[2]]]]), cv2.COLOR_HSV2RGB)[0][0]
                self.color_label.config(text=f"Màu trung bình: RGB={rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}")

                cv2.circle(self.image, (center_x, center_y + int(center_y * 0.1)), 5, (0, 255, 0), 2)
                # Vẽ viền box
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.mask[y1:y2, x1:x2] = 255
                
        
        self.update_color(None)
    def update_color(self, event):
        if self.image is None or self.mask is None:
            return
            
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        for box in self.results_yolo[0].boxes:
            if box.conf[0] > 0.7:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Tìm tâm của hộp
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Lấy màu tại tâm
                center_color = hsv[center_y + int(center_y * 0.1), center_x]
                
                # Tạo khoảng màu rộng hơn để tìm kiếm vùng tối của vật thể
                lower_bound = np.array([max(0, center_color[0] - 30), 
                                      max(0, center_color[1] - 100),
                                      max(0, center_color[2] - 100)])
                upper_bound = np.array([min(180, center_color[0] + 30),
                                      min(255, center_color[1] + 100), 
                                      min(255, center_color[2] + 100)])
                
                 # Tạo mask dựa trên khoảng màu
                color_mask = cv2.inRange(hsv[y1:y2, x1:x2], lower_bound, upper_bound)
                
                # Áp dụng màu mới chỉ cho vùng có mask
                new_hue = int(self.hue_scale.get())
                new_sat = int(self.sat_scale.get())
                new_val = int(self.val_scale.get())
                
                region = hsv[y1:y2, x1:x2]

                blend_factor = 0.7 
                
                # Tính toán màu mới dựa trên pha trộn với màu gốc
                region[color_mask > 0, 0] = int(new_hue * blend_factor + region[color_mask > 0, 0].mean() * (1 - blend_factor))
                region[color_mask > 0, 1] = np.clip(int(new_sat * blend_factor + region[color_mask > 0, 1].mean() * (1 - blend_factor)), 0, 255)
                region[color_mask > 0, 2] = np.clip(int(new_val * blend_factor + region[color_mask > 0, 2].mean() * (1 - blend_factor)), 0, 255)

                # Hòa trộn với màu gốc
                blended_region = cv2.addWeighted(region, 0.7, hsv[y1:y2, x1:x2], 0.3, 0)
                hsv[y1:y2, x1:x2] = blended_region
                
                # Vẽ tâm và biên của vật thể
                cv2.circle(self.image, (center_x, center_y), 5, (0, 255, 0), -1)
                contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_shifted = contour + np.array([x1, y1])
                    cv2.drawContours(self.image, [contour_shifted], -1, (0, 0, 255), 2)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = HSVColorChanger(root)
    root.mainloop()