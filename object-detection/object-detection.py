import numpy as np
import cv2

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 8.0



MIN_MATCH_COUNT = 30
detector = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
flannParam = dict(algorithm = FLANN_INDEX_KDTREE, tree = 5)
flann = cv2.FlannBasedMatcher(flannParam,())

trainImage = cv2.imread('4box.JPG', 0)
trainKP, trainDescriptors = detector.detectAndCompute(trainImage, None)
trainImage2 = cv2.imread('drops.JPG',0)
tKP2, tDesc2 = detector.detectAndCompute(trainImage2, None)


# h, w = trainImage.shape
# trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
focalLength = (336 * KNOWN_DISTANCE) / KNOWN_WIDTH

cam = cv2.VideoCapture(0)

while True:
    ret, queryImageBGR = cam.read()
    queryImage = cv2.cvtColor(queryImageBGR, cv2.COLOR_BGR2GRAY)
    queryKP, queryDescriptors = detector.detectAndCompute(queryImage, None)
    matches = flann.knnMatch(queryDescriptors, trainDescriptors, 2)
    goodMatch1 = []
    goodMatch2 = []

    for m, n in matches:
        if(m.distance < 0.75*n.distance):
            goodMatch1.append(m)

    if(len(goodMatch1)>MIN_MATCH_COUNT):
        tp = []
        qp = []
        for m in goodMatch1:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp = np.float32((tp,qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImage.shape
        trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        marker = abs(queryBorder[3][0][0] - queryBorder[0][0][0])
        print("Markerrrrrr------")
        print(marker)
        inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker)
        cv2.putText(queryImageBGR, "%.2fft" % (inches / 12),(queryImageBGR.shape[1] - 250, queryImageBGR.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)

        # print(queryBorder)
        # print("-------------------")
        # print(queryBorder[0][0][0])
        # print("-------------------")
        # print(queryBorder[3][0])
        # print("-------------------")
        cv2.polylines(queryImageBGR, [np.int32(queryBorder)],True,(0,255,0),5, cv2.LINE_AA)
    else:
        print('Not enough matches')


    #   scanning for 2nd training image - drops
    # matches2 = flann.knnMatch(queryDescriptors, tDesc2, 2)
    # for m,n in matches2:
    #     if(m.distance < 0.75*n.distance):
    #         goodMatch2.append(m)
    # if(len(goodMatch2)>MIN_MATCH_COUNT):
    #     tp2 = []
    #     qp2 = []
    #     for m in goodMatch2:
    #         tp2.append(tKP2[m.trainIdx].pt)
    #         qp2.append(queryKP[m.queryIdx].pt)
    #     tp2,qp2 = np.float32((tp2,qp2))
    #     H, status = cv2.findHomography(tp2, qp2, cv2.RANSAC, 3.0)
    #     h, w = trainImage2.shape
    #     trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     queryBorder = cv2.perspectiveTransform(trainBorder, H)
    #     cv2.polylines(queryImageBGR, [np.int32(queryBorder)],True,(255,0,0),5, cv2.LINE_AA)
    # else:
    #     print('Not enough matches - 2')
# queryImageBGR = cv2.imread('2foot.jpeg')
# queryImage = cv2.cvtColor(queryImageBGR, cv2.COLOR_BGR2GRAY)
# queryKP, queryDescriptors = detector.detectAndCompute(queryImage, None)
# matches = flann.knnMatch(queryDescriptors, trainDescriptors, 2)
# goodMatch1 = []
# goodMatch2 = []
#
# for m, n in matches:
#     if(m.distance < 0.75*n.distance):
#         goodMatch1.append(m)
#
# if(len(goodMatch1)>MIN_MATCH_COUNT):
#     tp = []
#     qp = []
#     for m in goodMatch1:
#         tp.append(trainKP[m.trainIdx].pt)
#         qp.append(queryKP[m.queryIdx].pt)
#     tp,qp = np.float32((tp,qp))
#     H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
#     h, w = trainImage.shape
#     trainBorder = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     queryBorder = cv2.perspectiveTransform(trainBorder, H)
#     print(queryBorder)
#     cv2.polylines(queryImageBGR, [np.int32(queryBorder)],True,(0,255,0),5, cv2.LINE_AA)
    queryImageBGR = cv2.resize(queryImageBGR, (900, 800))
    cv2.imshow('result', queryImageBGR)
# cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
         break
