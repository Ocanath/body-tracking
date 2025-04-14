import math

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