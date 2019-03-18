import numpy as np
import cv2

detector = cv2.xfeatures2d.SIFT_create()
img = cv2.imread('design.JPG')

imgGray = cv2.imread('design.JPG',0)
kp, desc = detector.detectAndCompute(imgGray, None)

#imgKP = cv2.drawKeypoints(imgGray, kp, img, (0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,750)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
