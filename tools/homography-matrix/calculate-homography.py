import cv2 

img1 = cv2.imread("1.jpg", 0)
img2 = cv2.imread("2.jpg", 0)

patternSize = (8,6)

ret1, corners1 = cv2.findChessboardCorners(img1, patternSize)
ret2, corners2 = cv2.findChessboardCorners(img2, patternSize)

#print(corners1, corners2)

H, _ = cv2.findHomography(corners1, corners2)
print(H)

img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))

cv2.imshow('Warped Image', img1_warp)
cv2.waitKey(0)