from tkinter import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
from ultralytics import YOLO
from tkinter import filedialog, messagebox

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

        # Speed Detection Button
        img_speed = Image.open(r"D:\VSD\VDBTT1.png")
        img_speed = img_speed.resize((220, 220), Image.LANCZOS)
        self.photoimg_speed = ImageTk.PhotoImage(img_speed)

        self.b_speed = Button(self.bg_img, image=self.photoimg_speed, cursor="hand2", command=self.open_speed_page)
        self.b_speed.place(x=800, y=400, width=220, height=220)

        self.b_speed_label = Button(self.bg_img, text="Speed detection", font=("times new roman", 15, "bold"),
                                    bg="White", fg="Green", cursor="hand2", command=self.open_speed_page)
        self.b_speed_label.place(x=800, y=620, width=220, height=40)

    def open_object_page(self):
        self.new_window = Toplevel(self.root)
        self.app = SecondaryPage(self.new_window, "Object Detection Page") # type: ignore

    def open_speed_page(self):
        self.new_window = Toplevel(self.root)
        self.app = SpeedDetectionPage(self.new_window, "Speed Detection Page")


class SpeedDetectionPage:
    def __init__(self, root, title):
        self.root = root
        self.root.geometry("1920x1080+0+0")
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
        img_label.place(x=-1.5, y=-5)  # Adjust position as needed

        # Initialize video path and output path
        self.video_path = None
        self.output_path = None

        # Buttons for Speed Detection Actions
        upload_btn = Button(self.root, text="Upload Video", font=("times new roman", 15, "bold"),
                            bg="green", fg="white", cursor="hand2", command=self.upload_video)
        upload_btn.place(x=600, y=400, width=300, height=50)

        view_btn = Button(self.root, text="View Video", font=("times new roman", 15, "bold"),
                          bg="blue", fg="white", cursor="hand2", command=self.view_video)
        view_btn.place(x=1100, y=400, width=300, height=50)

        download_btn = Button(self.root, text="Download Video", font=("times new roman", 15, "bold"),
                               bg="red", fg="white", cursor="hand2", command=self.download_video)
        download_btn.place(x=600, y=600, width=300, height=50)

        back_btn = Button(self.root, text="Back to Main Page", font=("times new roman", 15, "bold"),
                          bg="gray", fg="white", cursor="hand2", command=self.root.destroy)
        back_btn.place(x=1100, y=600, width=300, height=50)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video", filetypes=(("Video Files", "*.mp4;*.avi"),))
        if not self.video_path:
            messagebox.showerror("Error", "No video selected!")
            return
        messagebox.showinfo("Success", "Video uploaded successfully!")

    def view_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first.")
            return
        
        # YOLOv8 Model
        model = YOLO('yolov8n.pt')
        PIXEL_TO_METER_RATIO = 0.05
        FRAME_RATE = 30
        vehicles = {}

        def calculate_speed(distance_pixels, time_seconds):
            distance_meters = distance_pixels * PIXEL_TO_METER_RATIO
            speed = distance_meters / time_seconds
            return speed * 3.6  # Convert speed to km/h

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open video file.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = "output_view_video.avi"
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), FRAME_RATE, (frame_width, frame_height))
        previous_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            time_elapsed = current_time - previous_time
            previous_time = current_time

            results = model(frame)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                classes = result.boxes.cls

                for i, box in enumerate(boxes):
                    class_id = int(classes[i].item())
                    if class_id not in [2, 3, 5, 7]:  # Filter for vehicles
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
                        speed = calculate_speed(distance_pixels, time_elapsed)
                        vehicles[vehicle_id]['speeds'].append(speed)
                        vehicles[vehicle_id]['position'] = (center_x, center_y)

                        avg_speed = np.mean(vehicles[vehicle_id]['speeds'][-10:])
                        cv2.putText(frame, f"ID: {vehicle_id} | Speed: {avg_speed:.2f} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output.write(frame)
            cv2.imshow("Speed Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('e'):
                break

        cap.release()
        output.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", "Video processed successfully. Press 'e' to close the view.")

    def download_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please upload a video first.")
            return
        
        self.output_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=(("AVI Files", "*.avi"),))
        if not self.output_path:
            messagebox.showerror("Error", "No location selected to save the video!")
            return
        
        # Processing the video to save the output
        self.process_video(self.output_path)

    def process_video(self, output_path):
        model = YOLO('yolov8n.pt')
        PIXEL_TO_METER_RATIO = 0.05
        FRAME_RATE = 30
        vehicles = {}

        def calculate_speed(distance_pixels, time_seconds):
            distance_meters = distance_pixels * PIXEL_TO_METER_RATIO
            speed = distance_meters / time_seconds
            return speed * 3.6  # Convert speed to km/h

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to open video file.")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), FRAME_RATE, (frame_width, frame_height))
        previous_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            time_elapsed = current_time - previous_time
            previous_time = current_time

            results = model(frame)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                classes = result.boxes.cls

                for i, box in enumerate(boxes):
                    class_id = int(classes[i].item())
                    if class_id not in [2, 3, 5, 7]:  # Filter for vehicles
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
                        speed = calculate_speed(distance_pixels, time_elapsed)
                        vehicles[vehicle_id]['speeds'].append(speed)
                        vehicles[vehicle_id]['position'] = (center_x, center_y)

                        avg_speed = np.mean(vehicles[vehicle_id]['speeds'][-10:])
                        cv2.putText(frame, f"ID: {vehicle_id} | Speed: {avg_speed:.2f} km/h", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            output.write(frame)

        cap.release()
        output.release()
        messagebox.showinfo("Success", f"Output video saved at {output_path}")
        

if __name__ == "__main__":
    root = Tk()
    obj = Har(root)
    root.mainloop()
