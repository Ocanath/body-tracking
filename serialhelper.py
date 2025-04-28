import struct
from PPP_stuffing import PPP_stuff
import serial
from serial.tools import list_ports
from calibration_helper import pixel_to_direction_vector
import math
from sauron_ik import get_ik_angles_double


def fletchers_checksum16(data: bytes) -> int:
    """Calculate Fletcher's checksum for 16-bit words"""
    sum1 = 0
    sum2 = 0
    
    # Process data in 16-bit chunks
    for i in range(0, len(data), 2):
        if i + 1 < len(data):
            # Little-endian word construction
            word = data[i] | (data[i + 1] << 8)
            sum1 = (sum1 + word) & 0xFFFF
            sum2 = (sum2 + sum1) & 0xFFFF
    
    return sum2 & 0xFFFF


def create_sauron_position_payload(th1: int, th2: int) -> bytes:
    # Create initial payload
    payload = bytearray(12)
    
    # Write command (POSITION = 0xFA) and 0 byte
    payload[0] = 0xFA
    payload[1] = 0
    
    # Write the two 32-bit integers in little-endian format
    # '<ii' means little-endian 32-bit integers
    struct.pack_into('<ii', payload, 2, th1, th2)
    
    # Calculate checksum
    checksum = fletchers_checksum16(payload[:10])
    # Pack checksum in little-endian format
    struct.pack_into('<H', payload, 10, checksum)
    
    # Calculate final payload size
    
    # PPP stuff the payload
    stuffed_payload = PPP_stuff(payload)
    
    return stuffed_payload



def autoconnect_serial():
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
	return slist

def pixel_to_payload(pixel_x, pixel_y, camera_matrix, dist, g2c_offset, q_offset):
	direction_vector = pixel_to_direction_vector(pixel_x, pixel_y, camera_matrix)
	direction_vector[0] = -direction_vector[0]# track sign inversion
	direction_vector[1] = -direction_vector[1]
	direction_vector[2] = direction_vector[2]
	direction_vector = direction_vector * dist + g2c_offset
	x = direction_vector[0]  
	y = direction_vector[1]
	z = direction_vector[2]
	
	# Apply rotation and calculate IK angles
	xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
	yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
	xf = xr
	yf = yr
	zf = z
		
	theta1_rad, theta2_rad = get_ik_angles_double(xf, yf, zf)
	theta1 = int(theta1_rad*2**14)
	theta2 = int(theta2_rad*2**14)
	
	theta1 = theta1 + q_offset[0]
	theta2 = theta2 + q_offset[1]
	
	pld = create_sauron_position_payload(theta1, theta2)
	return pld