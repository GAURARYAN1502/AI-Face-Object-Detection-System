import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Make sure this file is downloaded

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- FACE DETECTION ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    # --- OBJECT DETECTION WITH YOLO ---
    results = model(frame)  # Runs YOLOv8 detection
    annotated_frame = results[0].plot()  # Automatically draws boxes

    # Merge face detection rectangles with YOLO frame
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Face", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the result
    cv2.imshow("🔍 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
