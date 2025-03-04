import io
from sys import argv
import numpy as np
import cv2 as cv
import time

def get_frame():
    if cap.isOpened():
        ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit(1)
    else:
        return ret, frame

def get_matches(img1, img2):

    # Initiate ORB detector
    orb = cv.ORB_create() #IDK what and ORB is?

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False) #Brute Force Matcher

    # # Match descriptors.
    # matches = bf.match(des1,des2)

    # # Sort by distance.
    # matches = sorted(matches, key = lambda x:x.distance)

    matches = bf.knnMatch(des1, des2, k=2)

    matches_vtb = []
    sorted_matches = sorted(matches, key = lambda matchlist: kp1[matchlist[0].queryIdx].pt)

    for n in sorted_matches:
        n = sorted(n, key = lambda m:m.distance)
        if len(n)>=2:
            if ((n[0].distance/n[1].distance) < (0.5)):
                matches_vtb.append(cv.DMatch(n[0].queryIdx,n[0].trainIdx,n[0].imgIdx,n[0].distance))   

    # Draw matches.
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches_vtb,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3S = cv.resize(img3, (1280, 360)) 
    cv.imshow(str(argv[1]),img3S)

    
if __name__ == "__main__":

    # Check Input
    if (len(argv) != 3):
        print("Incorrect number of arguments input.\n")
        print("Propper call is:")
        print("python3 Visual_Odometry.py [Input Video File] [Output File]\n")
    
    cap = cv.VideoCapture(argv[1])
    
    prev_ret, prev_frame = get_frame()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = get_frame()
        

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # cv.imshow('Test', gray)
        get_matches(prev_gray, gray)


        prev_gray = gray
        time.sleep(0.1)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()