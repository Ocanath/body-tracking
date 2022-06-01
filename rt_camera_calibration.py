import numpy as np
import cv2

cap = cv2.VideoCapture(0)

CHECKERBOARD = (6,9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None
num_iter = 0
key = 0
while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 	
	# Find the chess board corners
	# If desired number of corners are found in the image then ret = true
	ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
	
	if ret == True:
		corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
		if key==ord(' '):
			objpoints.append(objp)
			imgpoints.append(corners2)
			print('saved!')
			num_iter += 1

		frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)
		
		if(num_iter > 15):
			break

	cv2.imshow('frame',frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break


"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# fovx,fovy,focalLength,principalPoint,aspectRatio = cv2.calibrationMatrixValues(mtx, frame.shape[0:2], .700, .600)
# print("fovx: ", fovx)
# print("fovy: ", fovy)
# print("focal length: ", focalLength)
# print("principalPoint", principalPoint)
# print("aspectRatio: ", aspectRatio)

cap.release()
cv2.destroyAllWindows()