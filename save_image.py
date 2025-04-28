import cv2
import os
from datetime import datetime

def main():
    # Create directory if it doesn't exist
    if not os.path.exists('calibration-images'):
        os.makedirs('calibration-images')
    
    # Open camera
    cap = cv2.VideoCapture(2)  # Adjust camera index if needed
    
    print("Press spacebar to capture an image")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # Save image on spacebar
        if key == ord(' '):
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"calibration-images/image_{timestamp}.png"
            
            # Save the image
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            
        # Quit on 'q'
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 