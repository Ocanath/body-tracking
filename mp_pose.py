import cv2
import mediapipe as mp
import numpy as np
import json
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from scipy import signal
from rtfilt import *
from vect_tools import *
import time
from serialhelper import create_sauron_position_payload, autoconnect_serial
from sauron_ik import get_ik_angles_double
import math
from calibration_helper import load_camera_calibration, pixel_to_direction_vector

"""
fx 	0 	cx
0 	fy 	cy
0	0	1

cx, cy is optical center
fx, fy is focal length

fx=fy on most cameras
"""

lpf_fps_sos = signal.iirfilter(2, Wn=0.7, btype='lowpass', analog=False, ftype='butter', output='sos', fs=30)	#filter for the fps counter

slist = autoconnect_serial()
ser = slist[0]

# For webcam input:
cap = cv2.VideoCapture(3)
with mp_pose.Pose(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as pose:
	
	tprev = cv2.getTickCount()	
	warr_fps = [0,0,0]
	
	success, image = cap.read()
	print("imsize: ", image.shape[0:2])
	
	def normalized_to_pixel_coords(normalized_coords, image_width, image_height):
		"""
		Convert MediaPipe normalized coordinates to pixel coordinates.
		
		Args:
			normalized_coords: A tuple of (x, y, z) where x,y are in [0,1] and z is depth
			image_width: Width of the image in pixels
			image_height: Height of the image in pixels
		
		Returns:
			Tuple of (pixel_x, pixel_y) coordinates
		"""
		x, y, z = normalized_coords
		pixel_x = int(x * image_width)
		pixel_y = int(y * image_height)
		return pixel_x, pixel_y

	# Load camera calibration data
	try:
		camera_matrix, dist_coeffs = load_camera_calibration('camera_calibration.json')
		print("Successfully loaded camera calibration data")
		print("fx = ", camera_matrix[0,0])
		print("fy = ", camera_matrix[1,1])
		print("cx = ", camera_matrix[0,2])
		print("cy = ", camera_matrix[1,2])
	except Exception as e:
		print(f"Error loading calibration data: {e}")
		print("Using default camera parameters")

	while cap.isOpened():
	
		ts = cv2.getTickCount()
		tdif = ts-tprev
		tprev = ts
		fps = cv2.getTickFrequency()/tdif
		pixel_x = 0
		pixel_y = 0
	
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		# Undistort the image using calibration data
		image = cv2.undistort(image, camera_matrix, dist_coeffs)

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = pose.process(image)
		if results.pose_landmarks:
	
			# Get image dimensions
			image_height, image_width = image.shape[:2]
			
			rshoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
			lshoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
			lhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
			rhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
			nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

			
			vis = np.array([rshoulder.visibility, lshoulder.visibility, lhip.visibility, rhip.visibility])
			vis = np.where(vis > 0.5, 1, 0)
			num_elements = np.sum(vis)
			
			rshoulder = to_vect(rshoulder)
			lshoulder = to_vect(lshoulder)
			lhip = to_vect(lhip)
			rhip = to_vect(rhip)
			


		
			m1 = np.c_[rshoulder, lshoulder, lhip, rhip]
			center_mass = m1.dot(vis)/num_elements
			cm4 = np.append(center_mass,1)#for R4 expression of a coordinate

			# Convert center of mass to pixel coordinates
			pixel_x, pixel_y = normalized_to_pixel_coords(
				(cm4[0], cm4[1], cm4[2]),
				image_width,
				image_height
			)
			# pixel_x, pixel_y = normalized_to_pixel_coords(
			# 	(nose.x, nose.y, nose.z),
			# 	image_width,
			# 	image_height
			# )

			# Convert pixel coordinates to real-world direction vector
			direction_vector = pixel_to_direction_vector(pixel_x, pixel_y, camera_matrix)
			
			# Print the direction vector
			# print(f"Direction vector: {direction_vector}")
			
			# Use the direction vector for your application
			x = -direction_vector[0]	#track sign inversion
			y = -direction_vector[1]
			z = direction_vector[2]
			xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
			yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
			theta1_rad, theta2_rad = get_ik_angles_double(xr, yr, z)
			theta1 = int(theta1_rad*2**14)
			theta2 = int(theta2_rad*2**14)
			pld = create_sauron_position_payload(theta1, theta2)
			ser.write(pld)

			# print(x, y, z, theta1*180./(2**14*math.pi), theta2*180./(2**14*math.pi))

	
			# l_list = landmark_pb2.NormalizedLandmarkList(
			# 	landmark = [
			# 		v4_to_landmark(cm4),
			# 		nose
			# 	]
			# )
			# mp_drawing.draw_landmarks(
			# 	image,
			# 	l_list,
			# 	[],
			# 	mp_drawing_styles.get_default_hand_landmarks_style(),
			# 	mp_drawing_styles.get_default_hand_connections_style())

			# hw_b = ht_rotz(0)
			# # hw_b[0:3,3] = np.array([0,-100e-3,0])
			# #print(center_mass[2])
			# center_mass = center_mass + np.array([-0.5, -0.5, 1.5])
			
			

		# # Draw the pose annotation on the image.
		# image.flags.writeable = True
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


			
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.pose_landmarks,
		# 	mp_pose.POSE_CONNECTIONS,
		# 	landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
		
		# mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
		
		# if 'pixel_x' in locals() and 'pixel_y' in locals():
		# 	cv2.drawMarker(image, (pixel_x, pixel_y), (0, 255, 0), 
		# 				 markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)


		# Note: imshow and waitKey are blocking and can have performance impacts.
		#remove them if you want to run at max speed
		cv2.imshow('MediaPipe Pose', image)

		if cv2.waitKey(5) & 0xFF == 27:
			break
			
		fpsfilt, warr_fps = py_sos_iir(fps, warr_fps, lpf_fps_sos[0])
		# print (fpsfilt, center_mass)
			
cap.release()