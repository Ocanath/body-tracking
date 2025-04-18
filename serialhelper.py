import struct
from PPP_stuffing import PPP_stuff

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
