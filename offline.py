import cv2 as cv
import numpy as np
import sys
import glob


squaresize = 25

def manualCornerDetection(img, size):
    return ret, corners

def calibrateCamera(size, imagenames):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

    # scale the objp to the size of the chessboard squares in mm:
    objp = objp * squaresize

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in imagenames:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, size, None)
    
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv.drawChessboardCorners(img, size, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(0)
            
    cv.destroyAllWindows()



images = glob.glob('*.jpg')
calibrateCamera((9,6), images)
