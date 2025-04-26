from ultralytics import YOLO
import cv2

# 1) Load the nano model
model = YOLO('yolov8n.pt')

# 2) Open your camera
cap = cv2.VideoCapture(3)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3) Detect only cats (COCO class id 15), streaming for speed
    results = model.predict(frame, classes=[15], stream=True)

    # 4) Draw bboxes & confidences
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Cat {conf:.05f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

    # 5) Show it
    cv2.imshow('Real-time Cat Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
