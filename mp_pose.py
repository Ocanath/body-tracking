import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from scipy import signal
from rtfilt import *
from vect_tools import *



lpf_fps_sos = signal.iirfilter(2, Wn=0.7, btype='lowpass', analog=False, ftype='butter', output='sos', fs=30)	#filter for the fps counter

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as pose:
	
	tprev = cv2.getTickCount()	
	warr_fps = [0,0,0]

	while cap.isOpened():
	
		ts = cv2.getTickCount()
		tdif = ts-tprev
		tprev = ts
		fps = cv2.getTickFrequency()/tdif

	
		success, image = cap.read()
		if not success:
			print("Ignoring empty camera frame.")
			# If loading a video, use 'break' instead of 'continue'.
			continue

		# To improve performance, optionally mark the image as not writeable to
		# pass by reference.
		image.flags.writeable = False
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		results = pose.process(image)
		if results.pose_landmarks:
	
			rshoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
			lshoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
			lhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
			rhip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

			vis = np.array([rshoulder.visibility, lshoulder.visibility, lhip.visibility, rhip.visibility])
			# vis = np.array([1,1,1,1])
			
			rshoulder = to_vect(rshoulder)
			lshoulder = to_vect(lshoulder)
			lhip = to_vect(lhip)
			rhip = to_vect(rhip)
			
		
		
			m1 = np.c_[rshoulder, lshoulder, lhip, rhip]
			center_mass = m1.dot(vis)/4			
			cm4 = np.append(center_mass,1)#for R4 expression of a coordinate
			

			# print(center_mass, vis)
			l_list = landmark_pb2.NormalizedLandmarkList(
				landmark = [
					v4_to_landmark(cm4)
				]
			)
			mp_drawing.draw_landmarks(
				image,
				l_list,
				[],
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())

			
			
			
			

		# Draw the pose annotation on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		mp_drawing.draw_landmarks(
			image,
			results.pose_landmarks,
			mp_pose.POSE_CONNECTIONS,
			landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

		if cv2.waitKey(5) & 0xFF == 27:
			break
			
		fpsfilt, warr_fps = py_sos_iir(fps, warr_fps, lpf_fps_sos[0])
		# print (fpsfilt, center_mass)
			
cap.release()