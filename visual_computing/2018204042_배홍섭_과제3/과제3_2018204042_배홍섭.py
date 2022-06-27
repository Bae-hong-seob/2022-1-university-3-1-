import numpy as np
import cv2 as cv
import math 

FLANN_INDEX_LSH    = 6

def matchKeypoints(kp1, kp2, descriptors1, descriptors2):

    flann_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2

    matcher = cv.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    # knn 매칭은 k=2 로 설정된 순위만큼 return. 즉 2번째로 가까운 매칭 결과까지 리턴
    # 따라서 rew_matches 에는 [1순위 매칭결과, 2순위 매칭결과]
    raw_matches = matcher.knnMatch(descriptors1, descriptors2, k = 2) #2

    good_correspondences = []
    for m in raw_matches:
        # 첫번째로 가까운 값이 두번째로 가까운 값의 *0.79 distance보다 작다면 good_coreespondences 리스트에 추가.
        # 즉 feature 중 좋은 feature 선별
        if len(m) == 2 and m[0].distance < m[1].distance * 0.79:
            good_correspondences.append((m[0].trainIdx, m[0].queryIdx))


    if len(good_correspondences) >= 4:

        kp1 = np.float32([kp1[i] for (_, i) in good_correspondences])
        kp2 = np.float32([kp2[i] for (i, _) in good_correspondences])


        H, status = cv.findHomography(kp1, kp2, cv.RANSAC , 5.0)

        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d good_correspondences found, not enough for homography estimation' % len(good_correspondences))


    return good_correspondences, H, status


   
def drawMatches(image1, image2, kp1, kp2, matches, status):
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    img_matching_result = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")

    img_matching_result[0:h2, 0:w2] = image2
    img_matching_result[0:h1, w2:] = image1

    for ((trainIdx, queryIdx), s) in zip(matches, status):

        if s == 1:
            keyPoint2 = (int(kp2[trainIdx][0]), int(kp2[trainIdx][1]))
            keyPoint1 = (int(kp1[queryIdx][0]) + w2, int(kp1[queryIdx][1]))
            cv.line(img_matching_result, keyPoint1, keyPoint2, (0, 255, 0), 1)

    return img_matching_result



def blending(img1, img2): # img1 이 3, img2 이 2(중심)
    
    #--② 마스크 생성, 합성할 이미지 전체 영역을 255로 셋팅
    src = img2[:,300:,:] # 전경
    src = cv.add(src,-50) # 밝기조절
    dst = img2 # 배경
    mask = np.full_like(src, 255)
    
    #--③ 합성 대상 좌표 계산(img2의 중앙)
    height, width = src.shape[:2]
    center = (dst.shape[1] - width//2, dst.shape[0] - height//2)
    
    #--④ seamlessClone 으로 합성 
    normal_1 = cv.seamlessClone(img1, img2, mask, center, cv.NORMAL_CLONE)
    normal_2 = cv.seamlessClone(src, dst, mask, center, cv.NORMAL_CLONE)
    #mixed = cv.seamlessClone(img2, img1, mask, center, cv.MIXED_CLONE)

    #--⑤ 결과 출력
    cv.imshow('mask', src)
    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.imshow('normal_1', normal_1)
    cv.imshow('normal_2', normal_2)
    cv.waitKey(0)
    
    return normal_2
    

def main():
    
    img1 = cv.imread('set_tree1.jpg') 
    img2 = cv.imread('set_tree2.jpg') 
    img3 = cv.imread('set_tree3.jpg') 
    
    #gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    
    detector = cv.BRISK_create()
    kp2, descriptors2 = detector.detectAndCompute(gray2, None)
    kp3, descriptors3 = detector.detectAndCompute(gray3, None)
    print('img3 - %d features, img2 - %d features' % (len(kp3), len(kp2)))

    kp2 = np.float32([keypoint.pt for keypoint in kp2])
    kp3 = np.float32([keypoint.pt for keypoint in kp3])
    
    matches, H, status = matchKeypoints(kp3, kp2, descriptors3, descriptors2)

    img_matching_result = drawMatches(img3, img2, kp3, kp2, matches, status)
    
    point = [0,0,1]
    A = np.array(point).transpose()
    point = [img3.shape[1],0,1]
    B = np.array(point).transpose()
    point = [0,img3.shape[0],1]
    C = np.array(point).transpose()
    point = [img3.shape[1],img3.shape[0],1]
    D = np.array(point).transpose()

    matA = H.copy()
    matB = H.copy()
    matC = H.copy()
    matD = H.copy()
    
    AA =np.matmul(matA,A)
    BB =np.matmul(matB,B)
    CC =np.matmul(matC,C)
    DD =np.matmul(matD,D)
    

    conerX = []
    conerY = []
    
    conerX.append(AA[0]/AA[2])
    conerX.append(BB[0]/BB[2])
    conerX.append(CC[0]/CC[2])
    conerX.append(DD[0]/DD[2])
    
    conerY.append(AA[1]/AA[2])
    conerY.append(BB[1]/BB[2])
    conerY.append(CC[1]/CC[2])
    conerY.append(DD[1]/DD[2])
    conerY.append(img3.shape[0])
    
    middleBoundary = [[0,0], [img3.shape[1],0],[0,img3.shape[0]],[img3.shape[1],img3.shape[0]]]
    

    
    if min(conerY) < 0:
        dx = 0
        dy = math.ceil( min(conerY))
        mtrx = np.float32([[1, 0, dx],
                   [0, 1, -dy]])
        
        img3 = cv.warpAffine(img3, mtrx, (img3.shape[1]+dx, img3.shape[0]-dy) )
        img2 = cv.warpAffine(img2, mtrx, (img2.shape[1]+dx, img2.shape[0]-dy) )
        middleBoundary[0][1] = middleBoundary[0][1] - dy
        middleBoundary[1][1] = middleBoundary[1][1] - dy
        middleBoundary[2][1] = middleBoundary[2][1] - dy
        middleBoundary[3][1] = middleBoundary[3][1] - dy
        
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
        
        detector = cv.BRISK_create()
        kp2, descriptors2 = detector.detectAndCompute(gray2, None)
        kp3, descriptors3 = detector.detectAndCompute(gray3, None)

        kp2 = np.float32([keypoint.pt for keypoint in kp2])
        kp3 = np.float32([keypoint.pt for keypoint in kp3])
        
        matches, H, status = matchKeypoints(kp3, kp2, descriptors3, descriptors2)

        img_matching_result = drawMatches(img3, img2, kp3, kp2, matches, status)
        
        
        point = [0,0,1]
        A = np.array(point).transpose()
        point = [img3.shape[1],0,1]
        B = np.array(point).transpose()
        point = [0,img3.shape[0],1]
        C = np.array(point).transpose()
        point = [img3.shape[1],img3.shape[0],1]
        D = np.array(point).transpose()

        conerX = []
        conerY = []

        matA = H.copy()
        matB = H.copy()
        matC = H.copy()
        matD = H.copy()

        AA =np.matmul(matA,A)
        BB =np.matmul(matB,B)
        CC =np.matmul(matC,C)
        DD =np.matmul(matD,D)

        conerX.append(AA[0]/AA[2])
        conerX.append(BB[0]/BB[2])
        conerX.append(CC[0]/CC[2])
        conerX.append(DD[0]/DD[2])

        conerY.append(AA[1]/AA[2])
        conerY.append(BB[1]/BB[2])
        conerY.append(CC[1]/CC[2])
        conerY.append(DD[1]/DD[2])
        conerY.append(img2.shape[0])
        
        # warpPerspective 란 원근을 변환하여 평면처럼 보이게 하는 것
        result = cv.warpPerspective(img3, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
    else:
        result = cv.warpPerspective(img3, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
    
    mtrx = np.float32([[1, 0, 0],
                   [0, 1, 0]])
    img2 = cv.warpAffine(img2, mtrx, (result.shape[1], result.shape[0]) )
    
    resultN = cv.addWeighted(result, 0.5, img2, 0.5,0)
    
    for i in range((resultN.shape[0] * resultN.shape[1])):
        x = (i) % (resultN.shape[1]) # 지금 진행하는 픽셀의 행 번호
        y = (i) // resultN.shape[1] # 지금 진행한는 픽셀의 열 번호 
        if(sum(resultN[y][x]) < sum(img2[y][x])):
            resultN[y][x] = img2[y][x]
            
    for i in range((resultN.shape[0] * resultN.shape[1])):
        x = (i) % (resultN.shape[1]) # 지금 진행하는 픽셀의 행 번호
        y = (i) // resultN.shape[1] # 지금 진행한는 픽셀의 열 번호 
        if(sum(resultN[y][x]) < sum(result[y][x])):
            resultN[y][x] = result[y][x]
    
    
    beforeWidth = max(conerX)
    
    # cv.imshow('result1', resultN)
    # cv.imshow('matching result', img_matching_result)
# /////////////////////
    img2 = resultN
    
    img1 = cv.rotate(img1, cv.ROTATE_180)
    img2 = cv.rotate(img2, cv.ROTATE_180)

    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    
 
    detector = cv.BRISK_create()
    kp1, descriptors1 = detector.detectAndCompute(gray1, None)
    kp2, descriptors2 = detector.detectAndCompute(gray2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))


    
    kp1 = np.float32([keypoint.pt for keypoint in kp1])
    kp2 = np.float32([keypoint.pt for keypoint in kp2])
    


    matches, H, status = matchKeypoints(kp1, kp2, descriptors1, descriptors2)
    img_matching_result = drawMatches(img1, img2, kp1, kp2, matches, status)
    point = [0,0,1]
    A = np.array(point).transpose()
    point = [img1.shape[1],0,1]
    B = np.array(point).transpose()
    point = [0,img1.shape[0],1]
    C = np.array(point).transpose()
    point = [img1.shape[1],img1.shape[0],1]
    D = np.array(point).transpose()

    conerX = []
    conerY = []
    
    matA = H.copy()
    matB = H.copy()
    matC = H.copy()
    matD = H.copy()
    
    AA =np.matmul(matA,A)
    BB =np.matmul(matB,B)
    CC =np.matmul(matC,C)
    DD =np.matmul(matD,D)
    
    conerX.append(AA[0]/AA[2])
    conerX.append(BB[0]/BB[2])
    conerX.append(CC[0]/CC[2])
    conerX.append(DD[0]/DD[2])
    
    conerY.append(AA[1]/AA[2])
    conerY.append(BB[1]/BB[2])
    conerY.append(CC[1]/CC[2])
    conerY.append(DD[1]/DD[2])
    conerY.append(img1.shape[0])

    
    if min(conerY) < 0:
        dx = 0
        dy =math.floor( min(conerY))
        mtrx = np.float32([[1, 0, dx],
                   [0, 1, -dy]])
        img1 = cv.warpAffine(img1, mtrx, (img1.shape[1]+dx, img1.shape[0]-dy) )
        img2 = cv.warpAffine(img2, mtrx, (img2.shape[1]+dx, img2.shape[0]-dy) )
        
        gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        
        detector = cv.BRISK_create()
        kp1, descriptors1 = detector.detectAndCompute(gray1, None)
        kp2, descriptors2 = detector.detectAndCompute(gray2, None)
        print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

        kp1 = np.float32([keypoint.pt for keypoint in kp1])
        kp2 = np.float32([keypoint.pt for keypoint in kp2])
        
        matches, H, status = matchKeypoints(kp1, kp2, descriptors1, descriptors2)

        img_matching_result = drawMatches(img1, img2, kp1, kp2, matches, status)

        point = [0,0,1]
        A = np.array(point).transpose()
        point = [img1.shape[1],0,1]
        B = np.array(point).transpose()
        point = [0,img1.shape[0],1]
        C = np.array(point).transpose()
        point = [img1.shape[1],img1.shape[0],1]
        D = np.array(point).transpose()

        conerX = []
        conerY = []

        matA = H.copy()
        matB = H.copy()
        matC = H.copy()
        matD = H.copy()

        AA =np.matmul(matA,A)
        BB =np.matmul(matB,B)
        CC =np.matmul(matC,C)
        DD =np.matmul(matD,D)

        conerX.append(AA[0]/AA[2])
        conerX.append(BB[0]/BB[2])
        conerX.append(CC[0]/CC[2])
        conerX.append(DD[0]/DD[2])

        conerY.append(AA[1]/AA[2])
        conerY.append(BB[1]/BB[2])
        conerY.append(CC[1]/CC[2])
        conerY.append(DD[1]/DD[2])
        conerY.append(img2.shape[0])

        result2 = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
    else:
        result2 = cv.warpPerspective(img1, H,(math.ceil(max(conerX))  ,  math.ceil(max(conerY))))
   
    mtrx = np.float32([[1, 0, 0],
                   [0, 1, 0]])
    img2 = cv.warpAffine(img2, mtrx, (result2.shape[1], result2.shape[0]) )
    
    
    resultN = cv.addWeighted(result2, 0.5, img2, 0.5,0)
    for i in range((resultN.shape[0] * resultN.shape[1])):
        x = (i) % (resultN.shape[1]) # 지금 진행하는 픽셀의 행 번호
        y = (i) // resultN.shape[1] # 지금 진행한는 픽셀의 열 번호 
        if(sum(resultN[y][x]) < sum(img2[y][x])):
            resultN[y][x] = img2[y][x]
            
    for i in range((resultN.shape[0] * resultN.shape[1])):
        x = (i) % (resultN.shape[1]) # 지금 진행하는 픽셀의 행 번호
        y = (i) // resultN.shape[1] # 지금 진행한는 픽셀의 열 번호 
        if(sum(resultN[y][x]) < sum(result2[y][x])):
            resultN[y][x] = result2[y][x]
    
    
    result2 = cv.rotate(resultN, cv.ROTATE_180)
    
    dx = max(conerX) - beforeWidth
    
    middleBoundary[0][0] = middleBoundary[0][0] + dx
    middleBoundary[1][0] = middleBoundary[1][0] + dx
    middleBoundary[2][0] = middleBoundary[2][0] + dx
    middleBoundary[3][0] = middleBoundary[3][0] + dx
    
    intMiddleBoundary = np.array(middleBoundary, dtype=int)
    saveImage = result2[ intMiddleBoundary[0][1] +5:intMiddleBoundary[2][1] - 5,
                        intMiddleBoundary[0][0] + 5: intMiddleBoundary[1][0] - 5].copy() 
    
    gauImg = cv.GaussianBlur( result2[intMiddleBoundary[0][1] - 5:intMiddleBoundary[2][1]+ 5, 
                                      intMiddleBoundary[0][0] - 5 : intMiddleBoundary[1][0] + 5] , (5,5),0)
    
    result2[ intMiddleBoundary[0][1] - 5:intMiddleBoundary[2][1]+ 5, 
            intMiddleBoundary[0][0] - 5 : intMiddleBoundary[1][0] + 5] =  gauImg
    result2[ intMiddleBoundary[0][1] +5:intMiddleBoundary[2][1] - 5, 
            intMiddleBoundary[0][0] + 5: intMiddleBoundary[1][0] - 5 ] = saveImage
    cv.imwrite('Stitching image.jpg',result2)
    cv.imshow('Image Stitching', result2)
    cv.waitKey()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()