import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from scipy import signal
from rtfilt import *
from vect_tools import *
import time
import serial
from serial.tools import list_ports
from serialhelper import create_sauron_position_payload
from sauron_ik import get_ik_angles_double
import math

"""
fx 	0 	cx
0 	fy 	cy
0	0	1

cx, cy is optical center
fx, fy is focal length

fx=fy on most cameras
"""
cameraMatrix = np.array([[583.34552231,   0.,         307.58226975],
	[  0.,         584.08552834, 253.14697676],
	[  0.,           0.,           1.,        ]])

 
 
lpf_fps_sos = signal.iirfilter(2, Wn=0.7, btype='lowpass', analog=False, ftype='butter', output='sos', fs=30)	#filter for the fps counter



""" 
	Find a serial com port.
"""
com_ports_list = list(list_ports.comports())
port = []
slist = []
for p in com_ports_list:
	if(p):
		pstr = ""
		pstr = p
		port.append(pstr)
		print("Found:", pstr)
if not port:
	print("No port found")

for p in port:
	try:
		ser = []
		ser = (serial.Serial(p[0],'2000000', timeout = 0))
		slist.append(ser)
		print ("connected!", p)
		break
		# print ("found: ", p)
	except:
		print("failded.")
		pass
print( "found ", len(slist), "ports.")


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
	min_detection_confidence=0.5,
	min_tracking_confidence=0.5) as pose:
	
	tprev = cv2.getTickCount()	
	warr_fps = [0,0,0]
	
	success, image = cap.read()
	print("imsize: ", image.shape[0:2])
	
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
			
			t = time.time()
			x = math.sin(t)
			y = math.cos(t)+10
			xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
			yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
			theta1, theta2 = get_ik_angles_double(xr, yr, 100)
			theta1 = int(theta1*math.pi*2**14)
			theta2 = int(theta2*math.pi*2**14)
			pld = create_sauron_position_payload(theta1, theta2)
			ser.write(pld)


		

			print(center_mass)
			l_list = landmark_pb2.NormalizedLandmarkList(
				landmark = [
					v4_to_landmark(cm4),
					nose
				]
			)
			mp_drawing.draw_landmarks(
				image,
				l_list,
				[],
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())

			hw_b = ht_rotz(0)
			# hw_b[0:3,3] = np.array([0,-100e-3,0])
			#print(center_mass[2])
			center_mass = center_mass + np.array([-0.5, -0.5, 1.5])
			
			

		# Draw the pose annotation on the image.
		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		# mp_drawing.draw_landmarks(
		# 	image,
		# 	results.pose_landmarks,
		# 	mp_pose.POSE_CONNECTIONS,
		# 	landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
		
		# mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
		
		# Flip the image horizontally for a selfie-view display.
		cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))

		if cv2.waitKey(5) & 0xFF == 27:
			break
			
		fpsfilt, warr_fps = py_sos_iir(fps, warr_fps, lpf_fps_sos[0])
		# print (fpsfilt, center_mass)
			
cap.release()