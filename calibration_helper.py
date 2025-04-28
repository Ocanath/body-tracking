import numpy as np
import json

def load_camera_calibration(filename):
    """Load camera calibration parameters from a JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays
    camera_matrix = np.array(data['camera_matrix'])
    dist_coeffs = np.array(data['dist_coeffs'])
    
    return camera_matrix, dist_coeffs

def load_robot_calibration(filename):
    """Load robot calibration parameters from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data['theta1_offset'], data['theta2_offset']
    except FileNotFoundError:
        print(f"File {filename} not found, loading zeros")
        return 0, 0

def save_robot_calibration(filename, theta1_offset, theta2_offset):
    """Save robot calibration parameters to a JSON file."""
    data = {
        "theta1_offset": theta1_offset,
        "theta2_offset": theta2_offset
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Calibration parameters saved to {filename}")

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