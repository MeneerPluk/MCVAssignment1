import cv2 as cv
import numpy as np
import glob


squaresize = 25
img = None
objp = None

clicks = list()
def click_event(event, x, y, flags, params):
    """
    This function is a mouse event handler.
    The mouslocation gets saved when the user clicks on the image.
    It also draws a circle on the clicked coordinates.
    """
    global img
    if event == cv.EVENT_LBUTTONDOWN:
        drawCircle(x, y, True)
        clicks.append([x,y])


def drawCircle(x, y, show=False):
    """
    This function draws a circle with a cross in the middle on the x and y location that are given.
    There is an option to show the image in the display window.
    """
    global img
    cv.line(img, (x, y-4), (x, y+4), (0,0,255), 1)
    cv.line(img, (x-4, y), (x+4, y), (0,0,255), 1)
    cv.circle(img, (x, y), 8, (0,0,255), 1)
    if show:
        cv.imshow('img', img)


def manualCornerDetection(size):
    """
    This function is used to manually select the chessboard corners by clicking on the outer corners.
    You need to select the outer corners from left to right, top to bottom.
    The grid is then interpolated by first getting the perspective transform of the outer corner points and
    then perspective transforming the 3d grid points using the obtained perspective transformation matrix.
    """
    global img
    clicks.clear()
    cv.imshow('img', img)
    cv.setMouseCallback('img', click_event, img)
    
    # we need 4 corners, so wait...
    while len(clicks) < 4:
        cv.waitKey(25)

    # 4 outer corners of checkerboard
    checkCorners = [[0, 0], [(size[0]-1)*squaresize, 0], [0, (size[1]-1)*squaresize], [(size[0]-1)*squaresize, (size[1]-1)*squaresize]]

    # do math magic
    persMx = cv.getPerspectiveTransform(np.float32(checkCorners), np.float32(clicks))

    # get (2d!!) checkerboard array in correct shape
    chkPts = objp[:,0:2]
    chkPts = chkPts.reshape(size[0]*size[1], 1, 2)

    # apply math magic
    persCheck = cv.perspectiveTransform(chkPts, persMx)

    # reset mouse callback
    cv.setMouseCallback('img', lambda *args : None)
    return persCheck



def cameraCalibration(size, imagefnames, outfname):
    """
    This function is the main calibration script.
    It takes a size (nr_rowcorners, nr_collumncorners), list of image file paths to use for the calibration 
    and a file name to output the intrinsic camera matrix into.
    The automated corner selection of opencv is used first and if it fails our own manual selection interface takes over.
    The calibrated camera intrinsic matrix then gets saved to a file for later use.
    """
    global img
    global objp

    #-----------------------------------Corner finding part----------------------------------------------
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, a grid of the dimensions that are given by size
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)

    # scale the objp to the size of the chessboard squares in mm:
    objp = objp * squaresize

    # Arrays to store object points and image points from all the images:
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    for fname in imagefnames:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, size, None)
    
        if not ret:
            corners2 = manualCornerDetection(size)
            ret = True
        else:
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        
        objpoints.append(objp)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(250)

    cv.destroyAllWindows()

    #------------------------------------------Calibration part------------------------------------------
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # writing the camera intrinsic matrix to a file for later use:
    s = cv.FileStorage(outfname, cv.FileStorage_WRITE)
    s.write('K', mtx)
    s.release()




if __name__ == "__main__":
    images = glob.glob('Run1/*.jpg')
    cameraCalibration((9,6), images, "intrinsicmatrixRun1")
