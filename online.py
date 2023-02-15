import cv2 as cv
import numpy as np
import os
import glob

squaresize = 25

def loadCameraMatrixFromFile(filename):
    r = cv.FileStorage(filename, cv.FILE_STORAGE_READ)    
    mtx = r.getNode("K").mat()
    r.release()
    return mtx



def drawAxis(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    print(corner)
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 2)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 2)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 2)
    return img

def onlinePhaseImages(size, imagefnames, cameraMatrixfname):

    mtx = loadCameraMatrixFromFile(cameraMatrixfname)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)
    objp = objp * squaresize

    # 3D points of the axis line-ends:
    axis = np.float32([[3*squaresize,0,0], [0,3*squaresize,0], [0,0,-3*squaresize]]).reshape(-1,3)

    for fname in imagefnames:
        img = cv.imread(fname)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, size,None)

        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, None)
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, None)
            img = drawAxis(img,corners2,imgpts)
            cv.imshow('img',img)
            k = cv.waitKey(0) & 0xFF

            # use the key s to save the image with the axis and cube
            if k == ord('s'):
                # writes the picture with axis and cube to a .png in the current directory:
                cv.imwrite(os.path.basename(fname).split('.')[0]+'_save.png', img)
    cv.destroyAllWindows()

if __name__ == "__main__":
    images = glob.glob('test/*.jpg')
    onlinePhaseImages((9,6), images, "CamMatrixRun1")

