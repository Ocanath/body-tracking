import cv2
import numpy as np
import json
import apriltag
import math
from serialhelper import create_sauron_position_payload, autoconnect_serial
from sauron_ik import get_ik_angles_double


def load_calibration(filename):
    """Load camera calibration parameters from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    
    return camera_matrix, dist_coeffs

def main():
    slist = autoconnect_serial()

    # Load camera calibration data
    camera_matrix, dist_coeffs = load_calibration('camera_calibration.json')
    
    # Initialize AprilTag detector
    detector = apriltag.Detector()
    
    # Open camera
    cap = cv2.VideoCapture(3)  # Adjust this index if needed
    
    # Define tag size in meters (adjust this based on your actual tag size)
    tag_size = 0.07  # 7cm
    direction_vector = np.array([0, 0, 0])
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags
        detections = detector.detect(gray)
        
        for detection in detections:
            # Get the corners of the tag
            corners = detection.corners.astype(int)
            
            # Draw the tag outline
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
            
            # Get the center of the tag
            center = np.mean(corners, axis=0).astype(int)
            
            # Calculate pose
            object_points = np.array([
                [-tag_size/2, -tag_size/2, 0],
                [tag_size/2, -tag_size/2, 0],
                [tag_size/2, tag_size/2, 0],
                [-tag_size/2, tag_size/2, 0]
            ])
            
            # Solve PnP to get rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(
                object_points,
                detection.corners,
                camera_matrix,
                dist_coeffs
            )
            
            if ret:
                # Print the 3D position
                # print(f"Tag {detection.tag_id} position (x,y,z): {tvec.flatten()}")
                direction_vector = tvec.flatten()
                # Draw the center point
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)
                
                # Draw the coordinate axes
                axis_length = tag_size/2
                axis_points = np.float32([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length]
                ]).reshape(-1, 3)
                
                imgpts, _ = cv2.projectPoints(
                    axis_points,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs
                )
                
                imgpts = imgpts.astype(int)
                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 0, 0), 3)  # X axis
                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y axis
                cv2.line(frame, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0, 0, 255), 3)  # Z axis
        
        # Use the direction vector for your application
        x = -direction_vector[0]	#track sign inversion
        y = -direction_vector[1]
        z = direction_vector[2]
        print(x, y, z)


        xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
        yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
        try:    #NaN sometimes
            theta1_rad, theta2_rad = get_ik_angles_double(xr, yr, z)
            theta1 = int(theta1_rad*2**14)
            theta2 = int(theta2_rad*2**14)
            """
                Todo:
                Create a mechanism that uses some kind of user input
                (keys, mouse, etc) which adjusts an offset to theta1 and theta2
                before sending them out to the robot. This adjustment will correct
                for any misalignment in the gimbal obtained with manual 'eyeballed' 
                calibration. In order for this to work, the transformation from 
                sperical origin to camera focal origin must be obtained.
            """
            pld = create_sauron_position_payload(theta1, theta2)
            if(len(slist) != 0):
                slist[0].write(pld)
        except Exception as e:
            print(f"Error: {e}")
            pass

        # Display the frame
        cv2.imshow('AprilTag Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 