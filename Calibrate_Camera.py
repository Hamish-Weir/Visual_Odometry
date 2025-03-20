from sys import argv
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

def calibrate_camera(image_files, pattern_size, square_size):
    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image plane

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by square size

    for img_file in image_files:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    # Perform camera calibration
    obj_points = [np.array(pts, dtype=np.float32) for pts in obj_points]
    img_points = [np.array(pts, dtype=np.float32) for pts in img_points]


    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    R, _ = cv2.Rodrigues(rvecs[0])  # Convert rotation vector to rotation matrix
    t = tvecs[0]  # Translation vector

    # Construct extrinsic matrix [R | t]
    Rt = np.hstack((R, t))  # Shape (3,4)

    # Compute projection matrix P = K * [R | t]
    P = K @ Rt  # Matrix multiplication
    
    return K, P


if __name__ == "__main__":
# Example usage:
    # The square size of the checkerboard is 30mm. The board size is 11x7
    image_files = [(argv[1] + f) for f in listdir(argv[1]) if isfile(join(argv[1], f))]
    pattern_size = (11, 7)  # Number of inner corners in the chessboard
    square_size = float(0.03)  # Chessboard square size in meters
    K, P = calibrate_camera(image_files, pattern_size, square_size)
    print("Intrinsic Matrix:\n", K)
    print("Projection Matrix:\n", P)

    file = open("data/calibration.txt", "w")
    formatted_P_string = " ".join(f"{num:.12e}" for num in P.flatten())
    print(formatted_P_string)
    file.write(formatted_P_string + "\n")
    
