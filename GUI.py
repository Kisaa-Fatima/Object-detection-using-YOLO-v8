import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
from tkinter import scrolledtext

model = YOLO("best (1).pt")

# Tracking detected objects and their counts to ignore repetitions
detected_objects = {}

# My object dictionary
object_prices = {
    "pencil": 10,
    "eraser": 5,
    "sharpner": 8,
    "scale": 12
}

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        detect_objects(file_path)

# Detecting objects using the YOLO model
def detect_objects(image_path):
    global detected_objects
    image = cv2.imread(image_path)

    results = model.predict(source=image, stream=True)

    for result in results:
        boxes = result.boxes.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        # Drawing bounding boxes and labels on the image
        for box, cls, score in zip(boxes, classes, scores):

            x1, y1, x2, y2 = box[:4] 
            x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]

            label = model.names[int(cls)]
            confidence = round(score * 100, 2)

            # Updating the count for the detected object category
            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label} ({detected_objects[label]})"  # Include the count in the label
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - text_size[1] - 5
            cv2.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), (0, 255, 0), -1)
            cv2.putText(image, text, (text_x, text_y + text_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    display_image(image)
    
# Function to display the image with object detections
def display_image(image):
    global detected_objects
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)

    # Create a new window to display the image
    window = tk.Toplevel()
    window.title("Object Detection Result")

    # Creating label to display the image
    label = tk.Label(window, image=image)
    label.pack()

    # Creating a text box to display the detected objects and their counts
    text_box = scrolledtext.ScrolledText(window, width=40, height=10)
    text_box.pack()
    text_box.insert(tk.END, "Detected Objects:\n")
    total_bill = 0  
    for obj, count in detected_objects.items():
        if obj in object_prices:
            price = object_prices[obj]
            text_box.insert(tk.END, f"{obj}: {count} ---- Price = {count * price}\n")
            total_bill += count * price
        else:
            text_box.insert(tk.END, f"{obj}: {count} ---- Price not defined\n")
    text_box.insert(tk.END, f"Total Bill: ${total_bill}\n")

    window.mainloop()

root = tk.Tk()
root.title("Object Detection App")

open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

root.mainloop()
