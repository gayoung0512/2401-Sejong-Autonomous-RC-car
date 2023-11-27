import cv2 as cv
import numpy as np
import sys
import os

def matching(img_color):
    max = 0
    for i in os.listdir('./project_data/'):
        path = './project_data/' + i
        src = cv.imread(path, cv.IMREAD_COLOR)  # dataset
        src = cv.resize(src, (800, 800))
        if img_color is None or src is None:
            print('Image load failed!')
            sys.exit()

        feature = cv.ORB_create()
        #sift = cv.SIFT_create()
        # 특징점 검출 및 기술자 계산
        kp1, desc1 = feature.detectAndCompute(img_color, None)
        kp2, desc2 = feature.detectAndCompute(src, None)
        '''
        # 특징점 매칭(shift)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv.FlannBasedMatcher(index_params, search_params)

        matches1 = flann.knnMatch(desc1, desc2, k=2)
        '''
        matcher = cv.BFMatcher_create(cv.NORM_HAMMING)
        matches1 = matcher.knnMatch(desc1, desc2, 2)  # knnMatch로 특징점 2개 검출
        good_matches1 = []
        # ratio=0.25
        for m in matches1:  # matches는 두개의 리스트로 구성
            if m[0].distance / m[1].distance < 0.7:  # 임계점 0.7
                good_matches1.append(m)  # 저장
                #print(good_matches1)

        # print('# of good_matches:', len(good_matches1))
        if (len(good_matches1) > max):
            image = src
            max = len(good_matches1)
            max_good_matches = good_matches1
            kp_max = kp2
            max_path = path
    print(max_path)
    #cv.imshow('img', image)
    #cv.waitKey()
    #cv.destroyAllWindows()

lower_red1 = np.array([179,51,51])
lower_red2 = np.array([0,51,51])
lower_red3 = np.array([169,51,51])
upper_red1 = np.array([180,255,255])
upper_red2 = np.array([9,255,255])
upper_red3 = np.array([179,255,255])

lower_blue1 = np.array([101,30,30])
lower_blue2 = np.array([91,30,30])
lower_blue3 = np.array([91,30,30])
upper_blue1 = np.array([111,255,255])
upper_blue2 = np.array([101,255,255])
upper_blue3 = np.array([101,255,255])

cap = cv.VideoCapture(0)

while (True):
    # img_color = cv.imread('2.jpg')
    ret, img_color = cap.read()
    height, width = img_color.shape[:2]
    img_color = cv.resize(img_color, (width, height), interpolation=cv.INTER_AREA)

    # 원본 영상을 HSV 영상으로 변환합니다.
    img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask1 = cv.inRange(img_hsv, lower_blue1, upper_blue1)
    img_mask2 = cv.inRange(img_hsv, lower_blue2, upper_blue2)
    img_mask3 = cv.inRange(img_hsv, lower_blue3, upper_blue3)
    img_mask4 = cv.inRange(img_hsv, lower_red1, upper_red1)
    img_mask5 = cv.inRange(img_hsv, lower_red2, upper_red2)
    img_mask6 = cv.inRange(img_hsv, lower_red3, upper_red3)
    img_mask = img_mask1 | img_mask2 | img_mask3 | img_mask4 | img_mask5 | img_mask6

    kernel = np.ones((11, 11), np.uint8)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_OPEN, kernel)
    img_mask = cv.morphologyEx(img_mask, cv.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv.bitwise_and(img_color, img_color, mask=img_mask)

    numOfLabels, img_label, stats, centroids = cv.connectedComponentsWithStats(img_mask)

    for idx, centroid in enumerate(centroids):
        if stats[idx][0] == 0 and stats[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        #print(centerX, centerY)

        if area > 10000:
            cv.circle(img_color, (centerX, centerY), 10, (0, 0, 255), 10)
            cv.rectangle(img_color, (x, y), (x + width, y + height), (0, 0, 255))
            if x*y > 10000:
                try:
                    matching(img_color)
                except : cv2.error



    cv.imshow('img_color', img_color)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
