import cv2
import numpy as np
import time
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO


class Har:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1920x1080+0+0")
        self.root.title("Vehicle SD System")

        # Main Page Background
        img_background = Image.open(r"D:\VSD\VEHICLESD.png")  # Replace with the correct path
        img_background = img_background.resize((1920, 1080), Image.LANCZOS)
        self.photoimg_background = ImageTk.PhotoImage(img_background)

        self.bg_img = Label(self.root, image=self.photoimg_background)
        self.bg_img.place(x=0, y=0, width=1920, height=1080)

        # Object Detection Button
        img_object = Image.open(r"D:\VSD\VDBTT2.png")
        img_object = img_object.resize((220, 220), Image.LANCZOS)
        self.photoimg_object = ImageTk.PhotoImage(img_object)

        self.b_object = Button(self.bg_img, image=self.photoimg_object, cursor="hand2", command=self.open_object_page)
        self.b_object.place(x=550, y=400, width=220, height=220)

        self.b_object_label = Button(self.bg_img, text="Object detection", font=("times new roman", 15, "bold"),
                                     bg="White", fg="Green", cursor="hand2", command=self.open_object_page)
        self.b_object_label.place(x=550, y=620, width=220, height=40)

        # Speed Detection Button
        img_speed = Image.open(r"D:\VSD\VDBTT1.png")
        img_speed = img_speed.resize((220, 220), Image.LANCZOS)
        self.photoimg_speed = ImageTk.PhotoImage(img_speed)

        self.b_speed = Button(self.bg_img, image=self.photoimg_speed, cursor="hand2", command=self.open_speed_page)
        self.b_speed.place(x=1000, y=400, width=220, height=220)

        self.b_speed_label = Button(self.bg_img, text="Speed detection", font=("times new roman", 15, "bold"),
                                    bg="White", fg="Green", cursor="hand2", command=self.open_speed_page)
        self.b_speed_label.place(x=1000, y=620, width=220, height=40)

    def open_object_page(self):
        self.new_window = Toplevel(self.root)
        self.app = SecondaryPage(self.new_window, "Object Detection Page", "object_detection")

    def open_speed_page(self):
        self.new_window = Toplevel(self.root)
        self.app = SecondaryPage(self.new_window, "Speed Detection Page", "speed_detection")


class SecondaryPage:
    def __init__(self, root, title, mode):
        self.root = root
        self.root.geometry("1920x1080")
        self.root.title(title)

        # Add a title label
        title_label = Label(self.root, text=title, font=("times new roman", 25, "bold"), bg="white", fg="blue")
        title_label.pack(side=TOP, fill=X)

        img_path = r"D:\VSD\VEHICLESD.png"  # Replace with the correct path
        img = Image.open(img_path)
        img = img.resize((1920, 1080), Image.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)

        # Place the image in a Label
        img_label = Label(self.root, image=self.photoimg)
        img_label.place(x=-1.5, y=-5)

        # Buttons for Upload, View, and Download Video
        self.mode = mode
        self.input_video_path = None
        self.output_video_path = None

        upload_btn = Button(self.root, text="Upload Video", font=("times new roman", 15, "bold"),
                            bg="green", fg="white", cursor="hand2", command=self.upload_video)
        upload_btn.place(x=600, y=400, width=200, height=50)

        view_btn = Button(self.root, text="View Video", font=("times new roman", 15, "bold"),
                          bg="blue", fg="white", cursor="hand2", command=self.view_video)
        view_btn.place(x=1100, y=400, width=200, height=50)

        download_btn = Button(self.root, text="Download Video", font=("times new roman", 15, "bold"),
                               bg="red", fg="white", cursor="hand2", command=self.download_video)
        download_btn.place(x=600, y=600, width=200, height=50)

        back_btn = Button(self.root, text="Back to Main Page", font=("times new roman", 15, "bold"),
                          bg="gray", fg="white", cursor="hand2", command=self.root.destroy)
        back_btn.place(x=1100, y=600, width=200, height=50)

    def upload_video(self):
        self.input_video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        print(f"Video selected: {self.input_video_path}")

        self.output_video_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        print(f"Output video will be saved as: {self.output_video_path}")

    def view_video(self):
        if self.mode == "object_detection":
            self.process_video(self.object_detection)
        elif self.mode == "speed_detection":
            self.process_video(self.speed_detection)

    def download_video(self):
        if self.output_video_path:
            print(f"Downloading video to {self.output_video_path}")

    def process_video(self, detection_method):
        cap = cv2.VideoCapture(self.input_video_path)
        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))

        previous_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            time_elapsed = current_time - previous_time
            previous_time = current_time

            # Process detections based on mode
            if detection_method(frame, time_elapsed):
                output.write(frame)

            cv2.imshow('Processing Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        output.release()
        cv2.destroyAllWindows()

    def object_detection(self, frame, time_elapsed):
        model = YOLO('yolov8n.pt')
        coco_classes = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
        ]
        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = confidences[i]
                class_id = int(classes[i].item())
                label = f"{coco_classes[class_id]} ({confidence:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return True

    def speed_detection(self, frame, time_elapsed):
        model = YOLO('yolov8n.pt')
        PIXEL_TO_METER_RATIO = 0.05
        vehicles = {}
        results = model(frame)
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls
            for i, box in enumerate(boxes):
                class_id = int(classes[i].item())
                if class_id not in [2, 3, 5, 7]:
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                vehicle_id = None
                for vid, pos in vehicles.items():
                    px, py = pos['position']
                    if abs(px - center_x) < 50 and abs(py - center_y) < 50:
                        vehicle_id = vid
                        break

                if vehicle_id is None:
                    vehicle_id = len(vehicles) + 1
                    vehicles[vehicle_id] = {'position': (center_x, center_y), 'speeds': []}
                else:
                    distance_pixels = np.linalg.norm(np.array([center_x, center_y]) - np.array(vehicles[vehicle_id]['position']))
                    speed = distance_pixels * PIXEL_TO_METER_RATIO / time_elapsed
                    vehicles[vehicle_id]['speeds'].append(speed)
                    vehicles[vehicle_id]['position'] = (center_x, center_y)
                    avg_speed = np.mean(vehicles[vehicle_id]['speeds'][-10:])
                    cv2.putText(frame, f"ID: {vehicle_id} | Speed: {avg_speed:.2f} km/h", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return True


if __name__ == "__main__":
    root = Tk()
    app = Har(root)
    root.mainloop()
