import cv2 as cv
import numpy as np
import glob

mouseCord = np.zeros((2), np.int64)
img = None

#click_event handler
def click_event(event, x, y, flags, params):
    global mouseCord
    global img
    if event == cv.EVENT_LBUTTONDOWN:
        mouseCord[0] = x
        mouseCord[1] = y
        img1 = img.copy()
        cv.circle(img1,(x,y),5,(255,0,0),-1)
        cv.imshow('Display window', img1)

#calculates and returns the corner grid in the same form as findchessboardcorners does:
def interpolateGrid(cornerpoints, size):
    grid = np.zeros((size[1],size[0],2), np.int64)
    #setting the top(red) row of coordinates into the array:
    grid[0,0] = cornerpoints[0]
    grid[0,-1] = cornerpoints[1]
    dx_top = (cornerpoints[1,0] - cornerpoints[0,0])/(size[0]-1)
    dy_top = (cornerpoints[1,1] - cornerpoints[0,1])/(size[0]-1)
    for i in range(1,size[0]-1):
        grid[0,i] = grid[0,0] + [int(i * dx_top), int(i* dy_top)]
        cv.circle(img, grid[0,i], 5, (255,0,0),-1 )
        cv.imshow('Display window', img)
    return

#Interface for manual corners selection for when findchessboardcorners fails:
def manualCornerSelection(img, size):
    ret = False
    fourcorners = np.zeros((4,2), np.int64)
    for i in range(fourcorners.shape[0]):
        term = False
        while not term:
            cv.setMouseCallback('Display window', click_event)
            k = cv.waitKey(0)
            if k == 13: #k==enterkey
                cv.circle(img,mouseCord,5,(255,0,0),-1)
                fourcorners[i] = mouseCord
                term = True
    cv.setMouseCallback('Display window', lambda *args : None)
    cv.waitKey(0)
    
    #interpolate the grid from the four outer corners
    corners = interpolateGrid(fourcorners, size)
    cv.waitKey(0)
    ret = True
    return ret, corners


def getting_started_images():
    global img
    images = glob.glob("*.jpg")
    for imname in images:
        img = cv.imread(imname)
        cv.imshow("Display window", img)

        ret, corners = cv.findChessboardCorners(img, (9,6), None)
        #if no chessboard is found automatically call manualCornerSelection:
        if not ret:
            #ret, corners = manualCornerSelection(img,(9,6))
            print(imname, ret)

        #draw the chessboardCorners and wait for any key to continue to the next image:
        cv.drawChessboardCorners(img, (9,6), corners, ret)
        cv.imshow("Display window", img)
        k = cv.waitKey(0)

if __name__ == "__main__":
    getting_started_images()
