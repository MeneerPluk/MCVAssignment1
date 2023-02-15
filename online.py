import cv2 as cv
import numpy as np
import sys
import glob

def loadCameraMatrixFromFile(filename):
    r = cv.FileStorage("intrinsicmatrix", cv.FILE_STORAGE_READ)    
    mtx = r.getNode("K").mat()
    r.release()
    return mtx
