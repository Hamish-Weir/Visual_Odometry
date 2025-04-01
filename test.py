import numpy as np
import cv2

# Function to simulate camera calibration
def calibrate_camera_example(square_size):
    pattern_size = (3, 3)  # 3x3 checkerboard
    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image plane

    # Object points (real-world coordinates in 3D space, without scaling yet)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    
    # Scale object points by square_size
    objp *= square_size

    # Simulating image points (not based on real images here)
    img_points.append(np.array([[50, 50], [150, 50], [250, 50], [50, 150], [150, 150], [250, 150], [50, 250], [150, 250], [250, 250]], dtype=np.float32))
    obj_points.append(objp)

    # Dummy camera matrix and distortion coefficients (since we can't run calibration without real images)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)  # Example intrinsic matrix
    dist_coeffs = np.zeros(5)  # No distortion for simplicity

    # Simulating camera calibration (we only need K, distortion, rvecs, and tvecs)
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (640, 480), K, dist_coeffs)
    
    # Extract rotation matrix (R) and translation vector (t) from rvecs and tvecs
    R, _ = cv2.Rodrigues(rvecs[0])  # Convert rotation vector to rotation matrix
    t = tvecs[0]  # Translation vector

    # Projection matrix P = K * [R | t]
    Rt = np.hstack((R, t.reshape(-1, 1)))  # Shape (3, 4)
    P = K @ Rt  # Matrix multiplication

    return K, P

# Testing with two different square_sizes
square_size_1 = 1.0  # Example square size in meters
square_size_2 = 2.0  # Double the size of the squares

K1, P1 = calibrate_camera_example(square_size_1)
K2, P2 = calibrate_camera_example(square_size_2)

# Print the results
print("Intrinsic Matrix K (square_size = 1.0):")
print(K1)
print("\nProjection Matrix P (square_size = 1.0):")
print(P1)

