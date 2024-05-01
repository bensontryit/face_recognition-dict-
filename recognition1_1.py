import dlib, numpy

predictor = "shape_predictor_68_face_landmarks.dat"  #人臉68特徵點模型
recogmodel = "dlib_face_recognition_resnet_model_v1.dat"  #人臉辨識模型

detector = dlib.get_frontal_face_detector()  #偵測臉部正面
sp = dlib.shape_predictor(predictor)  #讀入人臉特徵點模型
facerec = dlib.face_recognition_model_v1(recogmodel)  #讀入人臉辨識模型

#取得人臉特徵點向量
def getFeature(imgfile):
    img = dlib.load_rgb_image(imgfile)
    dets = detector(img, 1)
    for det in dets:
        shape = sp(img, det)  #特徵點偵測
        feature = facerec.compute_face_descriptor(img, shape)  #取得128維特徵向量
        return numpy.array(feature)  #轉換numpy array格式

#判斷是否同一人 
def samePerson(pic1, pic2):
    feature1 = getFeature(pic1)
    feature2 = getFeature(pic2)
    dist = numpy.linalg.norm(feature1-feature2)  # 計算歐式距離,越小越像
    print("歐式距離={}".format(dist))
    if dist < 0.3: 
        print("{} 和 {} 為同一個人！".format(pic1, pic2))
    else:
        print("{} 和 {} 不是同一個人！".format(pic1, pic2))
    print()
    
samePerson("media\\LINE_ALBUM_111.jpg", "media\\LINE_ALBUM_11106.jpg")  #不同人
samePerson("media\\LINE_ALBUM_1125.jpg", "media\\LINE_ALBUM_1120610.jpg")  #同一人