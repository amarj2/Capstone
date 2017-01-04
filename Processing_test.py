import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('test.jpg')
img = cv2.medianBlur(img,5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = cv2.medianBlur(gray_image,5)
ret,th1 = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
th3 = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)

cv2.namedWindow('Binary', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Grey', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adaptive Mean', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adaptive Gaussian', cv2.WINDOW_NORMAL)
cv2.imshow('Binary',th1)
cv2.imshow('Original',img)
cv2.imshow('Grey',gray_image)
cv2.imshow('Adaptive Mean',th2)
cv2.imshow('Adaptive Gaussian',th3)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret,th2_new = cv2.threshold(th2,127,255,cv2.THRESH_BINARY)
ret,th3_new = cv2.threshold(th3,127,255,cv2.THRESH_BINARY)
th2_new = cv2.morphologyEx(th2_new, cv2.MORPH_OPEN, kernel)
th3_new = cv2.morphologyEx(th3_new, cv2.MORPH_OPEN, kernel)
th2_new = cv2.morphologyEx(th2_new, cv2.MORPH_CLOSE, kernel)
th3_new = cv2.morphologyEx(th3_new, cv2.MORPH_CLOSE, kernel)

cv2.namedWindow('Adaptive Mean', cv2.WINDOW_NORMAL)
cv2.namedWindow('Adaptive Gaussian', cv2.WINDOW_NORMAL)
cv2.imshow('Adaptive Mean',th2)
cv2.imshow('Adaptive Gaussian',th3)
cv2.namedWindow('Filtered Adaptive Mean', cv2.WINDOW_NORMAL)
cv2.namedWindow('Filtered Adaptive Gaussian', cv2.WINDOW_NORMAL)
cv2.imshow('Filtered Adaptive Mean',th2_new)
cv2.imshow('Filtered Adaptive Gaussian',th3_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

#vein_test.jpg & vein_test_2.jpg
'''feature = img[1200:3600,900:2000]
img2 = img - img
img2[1200:3600,900:2000] = feature'''

#vein_noise.jpg
feature = img[1500:3850,990:1750]
img2 = img - img
img2[1500:3850,990:1750] = feature

cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Feature', cv2.WINDOW_NORMAL)
cv2.imshow('Original',img)
cv2.imshow('Feature',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()






