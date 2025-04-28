import math
from calibration_helper import pixel_to_direction_vector
import numpy as np

# Base plane distance constant
BASE_PLANE_DISTANCE = 35.0 / 2.0 + 19.84

def get_ik_angles_double(vx: float, vy: float, vz: float) -> tuple[float, float]:
    """
    Calculate inverse kinematics angles for the Sauron Eye system.
    
    Args:
        vx: x component of the target vector
        vy: y component of the target vector
        vz: z component of the target vector
        
    Returns:
        tuple[float, float]: (theta1, theta2) angles in radians
        Returns (-1, -1) if the calculation fails
    """
    vx_pow2 = vx * vx
    vx_pow4 = vx_pow2 * vx_pow2
    vy_pow2 = vy * vy
    vz_pow2 = vz * vz
    vz_pow4 = vz_pow2 * vz_pow2

    # Calculate first operand
    operand = vx_pow4 + vx_pow2 * vy_pow2 + 2 * vx_pow2 * vz_pow2 + vy_pow2 * vz_pow2 + vz_pow4
    if operand < 0:
        return -1, -1
    
    # Calculate O2Targy_0
    O2Targy_0 = -BASE_PLANE_DISTANCE * vy * vz / math.sqrt(operand)
    
    # Recalculate operand for O2Targz_0
    operand = vx_pow4 + vx_pow2 * vy_pow2 + 2 * vx_pow2 * vz_pow2 + vy_pow2 * vz_pow2 + vz_pow4
    if operand < 0:
        return -1, -1
    
    # Calculate O2Targz_0
    O2Targz_0 = BASE_PLANE_DISTANCE * vx * vy / math.sqrt(operand) + BASE_PLANE_DISTANCE

    # Calculate theta2_s2
    operand = (BASE_PLANE_DISTANCE - O2Targz_0) / BASE_PLANE_DISTANCE
    if operand < -1 or operand > 1:
        return -1, -1
    theta2_s2 = -math.asin(operand)

    # Calculate theta1_s2
    operand = O2Targy_0 / (BASE_PLANE_DISTANCE * math.cos(theta2_s2))
    if operand < -1 or operand > 1:
        return -1, -1
    theta1_s2 = -math.asin(operand)

    # Calculate final angles
    theta1 = math.atan2(vx, vz)
    theta2 = theta1_s2

    return theta1, theta2





"""
This function takes in a pixel coordinate, a camera matrix, a distance, and a g2c offset.
For a parallel vector result, set the distance to 1 and the g2c offset to 0. This is equivalent to
an infinite distance to the target with a g2c offset corresponding to the correct camera position relative to the laser gimbal.

If you have an accurate distance to the target, set dist equal to that distance and use the g2c offset from the robot itself; that should be

"""
def pixel_to_thetas(pixel_x, pixel_y, camera_matrix, dist=1, g2c_offset=np.array([0, 0, 0]), q_offset=np.array([0, 0])):
	xyz_vector = pixel_to_direction_vector(pixel_x, pixel_y, camera_matrix)
	xyz_vector[0] = -xyz_vector[0]# track sign inversion
	xyz_vector[1] = -xyz_vector[1]
	xyz_vector[2] = xyz_vector[2]
	xyz_vector = xyz_vector * dist + g2c_offset
	x = xyz_vector[0]  
	y = xyz_vector[1]
	z = xyz_vector[2]
	
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
	
	return theta1, theta2

def xyz_to_thetas(xyz, q_offset=np.array([0, 0])):
	# Use the direction vector for your application
	offset=np.array([0, -69.12e-3, 23.06e-3])	#from the robot, with user defined offset
	x = -xyz[0]	+ offset[0]
	y = -xyz[1] + offset[1]
	z = xyz[2] + offset[2]
	xr = x*math.cos(math.pi/4) - y*math.sin(math.pi/4)
	yr = x*math.sin(math.pi/4) + y*math.cos(math.pi/4)
	theta1_rad, theta2_rad = get_ik_angles_double(xr, yr, z)
	theta1 = int(theta1_rad*2**14)
	theta2 = int(theta2_rad*2**14)
	theta1 = theta1 + q_offset[0]
	theta2 = theta2 + q_offset[1]
	return theta1, theta2

	