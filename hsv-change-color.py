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
            try:
                self.camera = cv2.VideoCapture(1, cv2.CAP_DSHOW)
                if not self.camera.isOpened():
                    raise Exception("Không thể mở camera")
                    
                self.is_camera_on = True
                self.camera_btn.configure(text="Tắt camera")
            except Exception as e:
                print(f"Lỗi khi mở camera: {str(e)}")
                self.camera = None
                self.is_camera_on = False
            
    def update(self):
        if self.is_camera_on and self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                self.image = frame
                self.display_image(self.image, self.original_label)
                self.create_mask()
        
        self.root.after(10, self.update)
        
    def create_mask(self):
        if self.image is None:
            return

        # Chuyển ảnh sang grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Áp dụng Canny edge detection với ngưỡng thấp hơn
        edges = cv2.Canny(blurred, 30, 150)
        
        # Thực hiện phép giãn nở để kết nối các cạnh
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        class_ranges = {'accessories': 0, 'bags': 1, 'clothing': 2, 'shoes': 3}
        selected_class = class_ranges[self.classification_combo.get()]  
        self.results_yolo = self.model.track(self.image, classes=selected_class)

        self.mask = np.zeros_like(gray, dtype=np.uint8)
        
        for box in self.results_yolo[0].boxes:
            if box.conf[0] > 0.5:  # Giảm ngưỡng tin cậy
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Lấy vùng cạnh trong box
                roi_edges = dilated[y1:y2, x1:x2]
                
                # Tìm contours từ cạnh
                contours, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Lọc contours theo diện tích
                min_area = 100  # Điều chỉnh ngưỡng diện tích tối thiểu
                filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
                
                # Vẽ và tô màu contours vào mask
                cv2.drawContours(self.mask[y1:y2, x1:x2], filtered_contours, -1, 255, -1)
                
                # Vẽ box và tâm
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(self.image, (center_x, center_y), 5, (0, 255, 0), -1)
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Vẽ contours lên ảnh gốc
                for contour in filtered_contours:
                    contour_shifted = contour + np.array([x1, y1])
                    cv2.drawContours(self.image, [contour_shifted], -1, (0, 0, 255), 2)
                
                # Hiển thị màu trung bình của vùng được chọn
                mask_roi = self.mask[y1:y2, x1:x2]
                if np.any(mask_roi > 0):
                    avg_color = cv2.mean(self.image[y1:y2, x1:x2], mask=mask_roi)
                    self.color_label.config(text=f"Màu trung bình: RGB={int(avg_color[2])}, {int(avg_color[1])}, {int(avg_color[0])}")
        
        self.update_color(None)
        
    def update_color(self, event):
        if self.image is None or self.mask is None:
            return
            
        hsv = cv2.cvtColor(self.image.copy(), cv2.COLOR_BGR2HSV)
        result = self.image.copy()
        
        for box in self.results_yolo[0].boxes:
            if box.conf[0] > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Lấy vùng mask trong box
                mask_roi = self.mask[y1:y2, x1:x2]
                
                # Áp dụng màu mới
                new_hue = int(self.hue_scale.get())
                new_sat = int(self.sat_scale.get())
                new_val = int(self.val_scale.get())
                
                # Tạo ảnh màu mới
                colored_mask = np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)
                colored_mask[:,:] = [new_hue, new_sat, new_val]
                
                # Chuyển đổi sang BGR
                colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_HSV2BGR)
                
                # Áp dụng mask và blend
                mask_3d = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR) / 255.0
                result[y1:y2, x1:x2] = (1 - mask_3d) * result[y1:y2, x1:x2] + mask_3d * colored_mask_bgr
                
                # Vẽ contours
                contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_shifted = contour + np.array([x1, y1])
                    cv2.drawContours(result, [contour_shifted], -1, (0, 0, 255), 2)
        
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