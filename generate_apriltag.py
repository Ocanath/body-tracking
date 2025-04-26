import cv2

# pick your marker dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_APRILTAG_36h11
)

marker_id   = 42    # which tag ID you want
marker_size = 200   # output image will be 200×200 pixels

# --- the new API call:
marker_img = cv2.aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    marker_size
)

# save or display
cv2.imwrite("tag36h11_id42.png", marker_img)
