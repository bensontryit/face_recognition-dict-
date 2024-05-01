import numpy as np
import cv2  #影象處理庫OpenCV
import dlib

predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   #構建68點特徵
detector = dlib.get_frontal_face_detector()  #偵測臉部正面

img = cv2.imread("media\\LINE_ALBUM_1120609.jpg")  #讀取影象
dets = detector(img, 1)  #偵測人臉
for det in dets:
    #人臉關鍵點識別
    landmarks = []
    for p in predictor(img, det).parts():  
        landmarks.append(np.matrix([p.x, p.y])) 
    # 取得68點座標
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])  #[0,0]為x坐標,[0,1]為y坐標
        cv2.circle(img, pos, 5, color=(0, 255, 0))  #畫出68個小圓點
        font = cv2.FONT_HERSHEY_SIMPLEX  # 利用cv2.putText輸出1-68
        #引數依次是：圖片，新增的文字，座標，字型，字型大小，顏色，字型粗細
        cv2.putText(img, str(idx+1), pos, font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)

cv2.namedWindow("img", 2)     
cv2.imshow("img", img)       #顯示影象
cv2.waitKey(0)
cv2.destroyWindow("img")