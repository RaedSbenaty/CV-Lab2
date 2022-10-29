import cv2
import numpy as np
import matplotlib.pyplot as plt


def affine():
    img = cv2.imread('files/img.png')
    rows, cols, depth = img.shape

    p1, p2, p3 = [0, 0], [170, 0], [0, 100]
    new_p1, new_p2, new_p3 = [200, 0], [300, 50], [150, 50]

    pts1 = np.array([p1, p2, p3]).astype(np.float32)
    pts2 = np.array([new_p1, new_p2, new_p3]).astype(np.float32)

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.line(img, p1, p2, (0, 255, 0), 6)
    cv2.line(img, p2, p3, (0, 255, 0), 6)
    cv2.line(img, p3, p1, (0, 255, 0), 6)

    cv2.line(img, new_p1, new_p2, (0, 0, 255), 6)
    cv2.line(img, new_p2, new_p3, (0, 0, 255), 6)
    cv2.line(img, new_p3, new_p1, (0, 0, 255), 6)

    cv2.line(dst, new_p1, new_p2, (0, 0, 255), 6)
    cv2.line(dst, new_p2, new_p3, (0, 0, 255), 6)
    cv2.line(dst, new_p3, new_p1, (0, 0, 255), 6)

    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


def perspective():
    img = cv2.imread('files/sudoku.png')
    rows, cols, depth = img.shape

    p1, p2, p3, p4 = [40, 52], [283, 42], [20, 307], [300, 310]
    new_p1, new_p2, new_p3, new_p4 = [0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]

    pts1 = np.array([p1, p2, p3, p4]).astype(np.float32)
    pts2 = np.array([new_p1, new_p2, new_p3, new_p4]).astype(np.float32)
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))

    cv2.circle(img, p1, 5, (0,  0, 255), -1)
    cv2.circle(img, p2, 5, (0,  0, 255), -1)
    cv2.circle(img, p3, 5, (0,  0, 255), -1)
    cv2.circle(img, p4, 5, (0,  0, 255), -1)

    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


affine()
perspective()
