import io
import os
from sys import argv
import numpy as np
import cv2 as cv
import time
from tqdm import tqdm

class Visual_Odometry():
    def __init__(self, input_file):
        self.orb = orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) #Brute Force Matcher
        self.threshold = 0.9
        self.images, self.videoLength = self._get_frames("data/video_frames")
        self.K, self.P = self._get_calibration("data/calibration.txt")
        self.orb = cv.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    
    @staticmethod
    def _get_calibration(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _Make_Transformation_Matrix(R,t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    @staticmethod 
    def _get_frames(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv.imread(path, cv.IMREAD_GRAYSCALE) for path in image_paths], len(image_paths)

    def Decompose_Essential_Matrix(self, q1, q2, E):
        #There are 4 different combinations of Translation * Rotation that can make any individual Essential Matrix
        #We are Deciding which Singular solution, among the Two Valid solutions, among the Four Possible solutions is Correct and Infront of the Camera
        R1, R2, t = cv.decomposeEssentialMat(E)
        T1 = self._Make_Transformation_Matrix(R1,np.ndarray.flatten(t))
        T2 = self._Make_Transformation_Matrix(R2,np.ndarray.flatten(t))
        T3 = self._Make_Transformation_Matrix(R1,np.ndarray.flatten(-t))
        T4 = self._Make_Transformation_Matrix(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))
        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv.triangulatePoints(self.P, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives)
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)

    def get_frame(self, i):
        return self.images[i]

    def get_matches(self, i):

        # Initiate ORB detector
        img1 = self.get_frame(i-1)
        img2 = self.get_frame(i)

        # find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1,None)
        kp2, des2 = self.orb.detectAndCompute(img2,None)

        # # Match descriptors.
        # matches = self.bf.match(des1,des2)

        # # Sort by distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        # Perform knn matching
        matches = self.flann.knnMatch(des1, des2, k=2)

        # Apply the ratio test to filter good matches (Lowe's ratio test)
        good_matches = []
        for m,n in [match for match in matches if len(match) == 2]:
            if m.distance < self.threshold*n.distance:
                good_matches.append(m)

        # Prepare the points for the matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # If you want to show the matches
        if show_matches:
            # Draw the matches
            img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Show the image with matches
            # Wait until a key is pressed
            img_matchesS = cv.resize(img_matches, (1280, 360)) 
            cv.imshow("Matches", img_matchesS)
            cv.waitKey(0)

        return q1, q2

    def get_pose(self, q1, q2):
        
        Essential, mask = cv.findEssentialMat(q1, q2, self.K)
        
        if show_essential:
           print ("\nEssential matrix:\n" + str(Essential))

        R, t = self.Decompose_Essential_Matrix(q1, q2, Essential)

        return self._Make_Transformation_Matrix(R,t)

    
if __name__ == "__main__":

    # Check Input
    if (len(argv) < 3):
        print("Incorrect number of arguments input.")
        print("     Propper call is:")
        print("     python3 Visual_Odometry.py [Input Video File] [Output File] -Flags")
        print("     Flags:")
        print("         -H       Help")
        print("         -M       Show Matches")
        exit(1)

    flags = argv[3:]
    
    if "-M" in flags:show_matches = True
    else:show_matches = False
    if "-V" in flags:show_video = True
    else:show_video = False
    if "-E" in flags:show_essential = True
    else:show_essential = False

    vo = Visual_Odometry(argv[1])

    if show_video:
        vo.play_video()

    # for i in range(vo.videoLength):
    #     q1, q2 = vo.get_matches(i)
    #     vo.get_pose(q1,q2)

    gt_path = []
    estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    file = open("data/path.txt", "w")
    for i in range(vo.videoLength):
        
        if i == 0:
            # cur_pose = gt_pose

            #Assume start is always looking forwards, can be manualy changed later, idc
            pose = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
            T = np.fromstring(pose, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            cur_pose = T
        else:
            q1, q2 = vo.get_matches(i)
            transformation = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transformation))
            # print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y,z: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) + "  " + str(cur_pose[1,3]) )
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3], cur_pose[1,3]))
        pstring = str(cur_pose[0, 3]) + " " + str(cur_pose[2, 3]) + " " + str(cur_pose[1, 3])
        file.write(pstring + "\n")
        
    # file = open("data/path.txt", "w")
    # for Est in estimated_path:
        # formatted_P_string = " ".join(f"{num:.12e}" for num in Est)
        # print(formatted_P_string)
        # file.write(formatted_P_string + "\n")
    file.close()
    
    # plotting.visualize_paths(estimated_path, estimated_path, "Visual Odometry", file_out=os.path.basename(argv[2]) + ".html")
    