# 🔍 AI Face & Object Detection System

This is a Python-based real-time face and object detection system that uses **OpenCV** for camera access and face detection, and **YOLOv8 (You Only Look Once)** for advanced object detection. The project captures live video feed from the webcam, detects human faces using Haarcascade, and simultaneously detects multiple objects like person, bottle, cell phone, laptop, etc. using a pre-trained YOLOv8 model.

---

## 🚀 Key Features

- ✅ Real-time detection using live webcam feed
- 💁 Detects human faces using Haarcascade classifier
- 📦 Detects 80+ objects (COCO dataset) like person, bottle, mobile, laptop, chair, etc.
- 🧠 Built using YOLOv8 for high-speed object detection
- 🔲 Draws bounding boxes with labels on detected items
- 🔧 Easy to modify and extend for your own use-case (like money or knife detection)

---

## 🎯 Real-World Applications

- 🔐 Surveillance & security monitoring
- 🏫 Classroom or lab attendance systems
- 🧠 AI vision learning projects for students & developers
- 💵 Currency or weapon detection (with custom training)
- 🤖 Smart robot vision (for detecting surroundings)

---

## 🧰 Tech Stack

| Tool / Library     | Purpose                          |
|--------------------|----------------------------------|
| Python             | Programming Language             |
| OpenCV             | Face Detection + Webcam Input    |
| YOLOv8 (Ultralytics)| Object Detection (Pre-trained) |
| Haarcascade        | Classical Face Detection Model   |

---

## 📦 Dependencies

### Required Libraries:
Install these using pip:
```bash
pip install opencv-python
pip install ultralytics

Download YOLOv8n weights 
yolo download model=yolov8n.pt

Run: python main.py
