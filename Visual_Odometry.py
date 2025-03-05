import io
from sys import argv
import numpy as np
import cv2 as cv
import time

class Visual_Odometry():
    def __init__(self, input_file):
        self.orb = orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) #Brute Force Matcher
        self.threshold = 0.5
        self.cap = cv.VideoCapture(input_file)

    def process_video(self):
        prev_frame = self.get_frame()
        prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

        while self.cap.isOpened():
            frame = self.get_frame()

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            self.get_matches(prev_gray, gray)


            prev_gray = gray
            time.sleep(0.1)
            
            if cv.waitKey(1) == ord('q'):
                break
        
        self.cap.release()
        cv.destroyAllWindows()


    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            exit(1)
        else:
            return frame

    def get_matches(self, img1, img2):

        # Initiate ORB detector
         #IDK what and ORB is?

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

        # # Draw matches.
        img3 = cv.drawMatches(img1,kp1,img2,kp2,matches_vtb,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3S = cv.resize(img3, (1280, 360)) 
        cv.imshow(str(argv[1]),img3S)

    
if __name__ == "__main__":

    # Check Input
    if (len(argv) != 3):
        print("Incorrect number of arguments input.\n")
        print("Propper call is:")
        print("python3 Visual_Odometry.py [Input Video File] [Output File]\n")
    
    vo = Visual_Odometry(argv[1])
    vo.process_video()