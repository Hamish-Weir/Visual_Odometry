import io
import os
from sys import argv
import numpy as np
import cv2 as cv
import time
from tqdm import tqdm
from lib.visualisation import plotting

class Visual_Odometry():
    def __init__(self, input_file):
        self.orb = orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) #Brute Force Matcher
        self.threshold = 0.5
        self.cap = cv.VideoCapture(input_file)
        self.videoLength = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.images = self.get_frames()


    def play_video(self):
        for i in range(0,self.videoLength):
            frame = self.get_frame(i)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            time.sleep(0.05)
            cv.imshow(str(argv[1]),frame)
            if cv.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()

    def get_frame(self, i):
        return self.images[i]
        
    def get_frames(self):
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return []
        frame_list = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop when the video ends
            frame_list.append(frame)
        self.cap.release()
        return frame_list


    def get_matches(self, i):

        # Initiate ORB detector
         #IDK what and ORB is?
        img1 = self.get_frame(i-1)
        img2 = self.get_frame(i)

        # find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(img1,None)
        kp2, des2 = self.orb.detectAndCompute(img2,None)

        # # Match descriptors.
        # matches = self.bf.match(des1,des2)

        # # Sort by distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        matches = self.bf.knnMatch(des1, des2, k=2)

        # Sort matches based on the keypoint positions in kp1
        sorted_matches = sorted(matches, key=lambda m: kp1[m[0].queryIdx].pt)

        matches_vtb = [
            cv.DMatch(m[0].queryIdx, m[0].trainIdx, m[0].imgIdx, m[0].distance)
            for m in sorted_matches if len(m) >= 2 and (m[0].distance / m[1].distance) < 0.5
        ]

        if show_matches:
            img3 = cv.drawMatches(img1,kp1,img2,kp2,matches_vtb,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            img3S = cv.resize(img3, (1280, 360)) 
            cv.imshow(str(argv[1]),img3S)

        return kp1, kp2

    def get_pose(self, kp1, kp2):
        
        pass
    
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

    vo = Visual_Odometry(argv[1])

    if show_video:
        vo.play_video()

    gt_path = []
    estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    for i in range(vo.videoLength):
        if i == 0:
            # cur_pose = gt_pose

            #Assume start straite, can be manualy changed later, idc
            pose = "1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00"
            T = np.fromstring(pose, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            cur_pose = T
        else:
            kp1, kp2 = vo.get_matches(i)
            transf = vo.get_pose(kp1, kp2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            # print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(argv[2]) + ".html")

    