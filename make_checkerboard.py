import cv2
import numpy as np
import argparse

def create_checkerboard(width, height, square_size, output_file):
    """
    Create a checkerboard pattern for camera calibration.
    
    Args:
        width: Number of squares horizontally
        height: Number of squares vertically
        square_size: Size of each square in pixels
        output_file: Output filename
    """
    # Create a white image
    img = np.ones((height * square_size, width * square_size), dtype=np.uint8) * 255
    
    # Draw black squares
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                y = i * square_size
                x = j * square_size
                img[y:y+square_size, x:x+square_size] = 0
    
    # Save the image
    cv2.imwrite(output_file, img)
    print(f"Checkerboard saved to {output_file}")
    print(f"Pattern size: {width}x{height} squares")
    print(f"Square size: {square_size} pixels")
    print(f"Total image size: {img.shape[1]}x{img.shape[0]} pixels")

def main():
    parser = argparse.ArgumentParser(description='Generate a checkerboard pattern for camera calibration')
    parser.add_argument('-w', '--width', type=int, default=9,
                        help='Number of squares horizontally (default: 9)')
    parser.add_argument('-t', '--height', type=int, default=6,
                        help='Number of squares vertically (default: 6)')
    parser.add_argument('-s', '--square-size', type=int, default=100,
                        help='Size of each square in pixels (default: 100)')
    parser.add_argument('-o', '--output', type=str, default='checkerboard.png',
                        help='Output filename (default: checkerboard.png)')
    
    args = parser.parse_args()
    
    create_checkerboard(args.width, args.height, args.square_size, args.output)

if __name__ == '__main__':
    main() 