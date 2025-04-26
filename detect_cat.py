from ultralytics import YOLO
import cv2
import json
import numpy as np
import math
from serialhelper import create_sauron_position_payload, autoconnect_serial
from sauron_ik import get_ik_angles_double
# 1) Load the nano model
model = YOLO('yolov8n.pt')


def load_calibration(filename):
	"""Load calibration parameters from a JSON file."""
	with open(filename, 'r') as f:
		data = json.load(f)
	
	# Convert lists back to numpy arrays
	camera_matrix = np.array(data['camera_matrix'])
	dist_coeffs = np.array(data['dist_coeffs'])
	
	return camera_matrix, dist_coeffs

def pixel_to_direction_vector(pixel_x, pixel_y, camera_matrix):
	"""
	Convert pixel coordinates to a real-world direction vector from the camera's optical center.
	
	Args:
		pixel_x, pixel_y: Pixel coordinates in the image
		camera_matrix: 3x3 camera matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
	
	Returns:
		direction_vector: 3D unit vector pointing from camera center to the point
	"""
	# Get camera parameters from matrix
	fx = camera_matrix[0,0]
	fy = camera_matrix[1,1]
	cx = camera_matrix[0,2]
	cy = camera_matrix[1,2]
	
	# Convert pixel coordinates to normalized image coordinates
	x = (pixel_x - cx) / fx
	y = (pixel_y - cy) / fy
	
	# Create direction vector (z=1 for normalized coordinates)
	direction = np.array([x, y, 1.0])
	
	# Normalize to unit vector
	direction = direction / np.linalg.norm(direction)
	
	return direction


try:
	camera_matrix, dist_coeffs = load_calibration('camera_calibration.json')
	print("Successfully loaded camera calibration data")
	print("fx = ", camera_matrix[0,0])
	print("fy = ", camera_matrix[1,1])
	print("cx = ", camera_matrix[0,2])
	print("cy = ", camera_matrix[1,2])
except Exception as e:
	print(f"Error loading calibration data: {e}")
	print("Using default camera parameters")


slist = autoconnect_serial()
# 2) Open your camera
cap = cv2.VideoCapture(3)
catxlocation = 10
catylocation = 10
direction_vector = np.array([0, 0, 0])
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3) Detect only cats (COCO class id 15), streaming for speed
    results = model.predict(frame, classes=[15], stream=True, verbose=False)

    # 4) Draw bboxes & confidences
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"Cat {conf:.5f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)
            catxlocation = int((x1 + x2) / 2)
            catylocation = int((y1 + y2) / 2)
            direction_vector = pixel_to_direction_vector(catxlocation, catylocation, camera_matrix)
            print(f"catpix = {catxlocation}, {catylocation}, tvec = {direction_vector}")

    cv2.drawMarker(frame, (catxlocation, catylocation), (0, 255, 0), 
                 markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    x = -direction_vector[0]    #track sign inversion
    y = -direction_vector[1]
    z = direction_vector[2]
    try:
        xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
        yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
        theta1_rad, theta2_rad = get_ik_angles_double(xr, yr, z)
        theta1 = int(theta1_rad*2**14)
        theta2 = int(theta2_rad*2**14)
        pld = create_sauron_position_payload(theta1, theta2)
        slist[0].write(pld)
    except Exception as e:
        print(f"Error calculating IK angles: {e}")

    # 5) Show it
    cv2.imshow('Real-time Cat Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
