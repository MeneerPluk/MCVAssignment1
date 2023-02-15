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



def drawAxisAndCube(img, corners, axisImgpts, cubeImgpts):
    # drawing the axis:
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(axisImgpts[0].ravel().astype(int)), (255,0,0), 3)
    img = cv.line(img, corner, tuple(axisImgpts[1].ravel().astype(int)), (0,255,0), 3)
    img = cv.line(img, corner, tuple(axisImgpts[2].ravel().astype(int)), (0,0,255), 3)

    # drawing the cube:
    cubeImgpts = np.int32(cubeImgpts).reshape(-1,2)
    # draw the bottom:
    img = cv.drawContours(img, [cubeImgpts[:4]],-1,(40,202,255),2)
    # draw the side lines:
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(cubeImgpts[i]), tuple(cubeImgpts[j]),(40,202,255),2)
    # draw top:
    img = cv.drawContours(img, [cubeImgpts[4:]],-1,(40,202,255),2)
    return img


def addCubeAndAxis2Frame(size, img, mtx, flag = None):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((size[0]*size[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:size[0],0:size[1]].T.reshape(-1,2)
    objp = objp * squaresize
    cubesize = 2*squaresize

    # 3D points of the axis line-ends:
    axis = np.float32([[3*squaresize,0,0], [0,3*squaresize,0], [0,0,-3*squaresize]]).reshape(-1,3)
    # 3D points of the cube corners:
    cube = np.float32([[0,0,0], [0,cubesize,0], [cubesize,cubesize,0], [cubesize,0,0],
                       [0,0,-cubesize],[0,cubesize,-cubesize],[cubesize,cubesize,-cubesize],
                       [cubesize,0,-cubesize] ])
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, size,None,flag)

    if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, None)
            # project 3D points to image plane
            axisImgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, None)
            cubeImgpts, jac = cv.projectPoints(cube, rvecs, tvecs, mtx, None)
            img = drawAxisAndCube(img,corners2,axisImgpts,cubeImgpts)

    return img

def onlinePhaseImages(size, imagefnames, cameraMatrixfname):
    mtx = loadCameraMatrixFromFile(cameraMatrixfname)

    for fname in imagefnames:
        img = cv.imread(fname)
        img = addCubeAndAxis2Frame(size, img, mtx)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF

        # use the key s to save the image with the axis and cube
        if k == ord('s'):
            # writes the picture with axis and cube to a .png in the current directory:
            cv.imwrite(os.path.basename(fname).split('.')[0]+'_save.png', img)
    cv.destroyAllWindows()


def onlinePhaseVideo(size, cameraMatrixfname):
    mtx = loadCameraMatrixFromFile(cameraMatrixfname)
    
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        img = addCubeAndAxis2Frame(size, img, mtx, cv.CALIB_CB_FAST_CHECK)
        cv.imshow('img',img)
        k = cv.waitKey(1)
        # use the key q to quit the program:
        if k == ord('q'):
            break
        
    

if __name__ == "__main__":
    #images = glob.glob('test/*.jpg')
    #onlinePhaseImages((9,6), images, "CamMatrixRun1")
    onlinePhaseVideo((9,6),"CamMatrixRun1")

