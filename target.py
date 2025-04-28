import cv2
import numpy as np
import math
from serialhelper import  autoconnect_serial, pixel_to_payload
from datetime import datetime
from calibration_helper import load_camera_calibration, load_robot_calibration, save_robot_calibration

# Global variables for mouse callback
mouse_x = 0
mouse_y = 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, mouse_clicked
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    elif event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_clicked = False

def main():
    slist = autoconnect_serial()

    # Load calibration data
    camera_matrix, dist_coeffs = load_camera_calibration('camera_calibration.json')
    user_input_theta1_offset, user_input_theta2_offset = load_robot_calibration('robot_calibration.json')
    print(f"Loaded robot offsets: {user_input_theta1_offset}, {user_input_theta2_offset}")
    
    # Open camera
    cap = cv2.VideoCapture(2)  # Adjust this index if needed
    
    # Create window and set mouse callback
    cv2.namedWindow('Target Tracking')
    cv2.setMouseCallback('Target Tracking', mouse_callback)
    
    # Initialize variables
    g2c_offset = np.array([0, 0, 0])	#if this is loaded with the correct offsets (63.12mm, 23.06mm) 
    target_distance = 1		#and this is the normalized distance from the camera to the target, then the laser should hit the target (in theory)
    toggle_print_pos = False
    
    print("Controls:")
    print("  - Move mouse to aim")
    print("  - Left click to lock/unlock target")
    print("  - 'q' to quit")
    print("  - 'p' to toggle position printing")
    print("  - 'l' to toggle theta locking")
    print("  - 't/g' to adjust x offset")
    print("  - 'y/h' to adjust y offset")
    print("  - 'u/j' to adjust z offset")
    print("  - 's' to save calibration")
    print("  - Space to save image")
    direction_vector = np.array([0, 0, 1])
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Undistort the image using calibration data
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Draw crosshair at mouse position
        cv2.drawMarker(frame, (mouse_x, mouse_y), (0, 255, 0), 
                      markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                
        try:
            pld = pixel_to_payload(mouse_x, mouse_y, camera_matrix, target_distance, g2c_offset, [user_input_theta1_offset, user_input_theta2_offset])
            if len(slist) != 0:
                slist[0].write(pld)
        except Exception as e:
            print(f"Error calculating IK angles: {e}")
        
        # Display the frame
        cv2.imshow('Target Tracking', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            toggle_print_pos = not toggle_print_pos
            print(f"Position printing: {'enabled' if toggle_print_pos else 'disabled'}")
        elif key == ord('t'):
            x_offset += 0.01
            print(f"x_offset: {x_offset}")
        elif key == ord('g'):
            x_offset -= 0.01
            print(f"x_offset: {x_offset}")
        elif key == ord('y'):
            y_offset += 0.01
            print(f"y_offset: {y_offset}")
        elif key == ord('h'):
            y_offset -= 0.01
            print(f"y_offset: {y_offset}")
        elif key == ord('u'):
            z_offset += 0.01
            print(f"z_offset: {z_offset}")
        elif key == ord('j'):
            z_offset -= 0.01
            print(f"z_offset: {z_offset}")
        elif key == ord('e'):
            user_input_theta1_offset = user_input_theta1_offset + 10
            print(f"theta1_offset: {user_input_theta1_offset}")
        elif key == ord('d'):
            user_input_theta1_offset = user_input_theta1_offset - 10
            print(f"theta1_offset: {user_input_theta1_offset}")
        elif key == ord('r'):
            user_input_theta2_offset = user_input_theta2_offset + 10
            print(f"theta2_offset: {user_input_theta2_offset}")
        elif key == ord('f'):
            user_input_theta2_offset = user_input_theta2_offset - 10
            print(f"theta2_offset: {user_input_theta2_offset}")
        elif key == ord('s'):
            save_robot_calibration("robot_calibration.json", user_input_theta1_offset, user_input_theta2_offset)
        elif key == ord(' '):
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration-images/image_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 