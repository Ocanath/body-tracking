#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera(images_path, pattern_size=(9,6), square_size=0.025):
    """
    Calibrate camera using checkerboard images.
    
    Args:
        images_path: Path to directory containing calibration images
        pattern_size: Tuple of (width, height) of inner corners
        square_size: Size of squares in meters (for real-world scale)
    
    Returns:
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size

    # Arrays to store object points and image points
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane

    # Get list of calibration images
    images = glob.glob(os.path.join(images_path, '*.jpg')) + glob.glob(os.path.join(images_path, '*.png'))
    
    if not images:
        raise ValueError(f"No images found in {images_path}")

    # Process each image
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner locations
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration(filename, camera_matrix, dist_coeffs):
    """Save calibration parameters to a file."""
    data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Calibration data saved to {filename}")

def load_calibration(filename):
    """Load calibration parameters from a file."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['camera_matrix'], data['dist_coeffs']

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Camera calibration using checkerboard pattern')
    parser.add_argument('images_path', help='Path to directory containing calibration images')
    parser.add_argument('--output', default='camera_calibration.pkl',
                      help='Output file for calibration parameters')
    parser.add_argument('--pattern-size', type=int, nargs=2, default=[9,6],
                      help='Number of inner corners in the checkerboard pattern (width, height)')
    parser.add_argument('--square-size', type=float, default=0.025,
                      help='Size of squares in meters')
    
    args = parser.parse_args()
    
    try:
        camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
            args.images_path, tuple(args.pattern_size), args.square_size)
        
        print("\nCalibration Results:")
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
        
        save_calibration(args.output, camera_matrix, dist_coeffs)
        
    except Exception as e:
        print(f"Error during calibration: {e}")

if __name__ == '__main__':
    main()
