import cv2
import numpy as np

def main():
    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    
    # Define minimum area for valid detections (in pixels)
    MIN_AREA = 50  # Adjust this based on your dot size
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for red color in HSV
        # Note: Red is at both ends of the HSV spectrum
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up the mask using morphological operations
        kernel = np.ones((3,3), np.uint8)
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw centroids for each contour that meets size criteria
        for contour in contours:
            # Calculate the area of the contour
            # area = cv2.contourArea(contour)
            
            # Only process contours larger than minimum area
            # if area > MIN_AREA:
                # Calculate moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                # Calculate centroid
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Draw a dot at the centroid
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                # Optional: Draw the contour and display area
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                # cv2.putText(frame, f"Area: {int(area)}", (cx-20, cy-20),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display the results
        cv2.imshow('Red Object Centroid Detection', frame)
        cv2.imshow('Red Mask', mask)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 