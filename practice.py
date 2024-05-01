import dlib
import numpy as np
import os

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
        return np.array(feature)  #轉換numpy array格式

#判斷是否同一人 
def samePerson(pic1, pic2):
    feature1 = getFeature(pic1)
    feature2 = getFeature(pic2)
    dist = np.linalg.norm(feature1-feature2)  # 計算歐式距離,越小越像
    print("歐式距離={}".format(dist))
    if dist < 0.3: 
        print("{} 和 {} 為同一個人！".format(pic1, pic2))
    else:
        print("{} 和 {} 不是同一個人！".format(pic1, pic2))
    print()
#遍歷整個dataset
def list_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

def encoding_dataset(jpg_files):
    container = np.zeros([len(jpg_files), 128])
    for i, jpg in enumerate(jpg_files):
        container[i,:] = getFeature(jpg)
    return container
def name_dictionary(jpg_files, container):
    name_dict = dict(zip(jpg_files, container))
    return name_dict

def in_dataset(unknown_pic, container):
    unknown_encoding = getFeature(unknown_pic)
    count = 0
    for i, dataset_jpg_encoding in enumerate(container):
        dist = np.linalg.norm(unknown_encoding-dataset_jpg_encoding)
        if dist < 0.4:
            print("The person is in the dataset")
            print(dist)
            print(jpg_files[i])
            count = 1
            break
        else:
            i = i + 1
            continue
    if(count == 0):
        print("The person is not in the dataset.")



dataset_directory = 'small_dataset'
jpg_files = list_jpg_files(dataset_directory)
encoding_container = encoding_dataset(jpg_files)
name_dict = name_dictionary(jpg_files, encoding_container)
print(name_dict)
unknown = 'unknown2.jpg'
recogition_process = in_dataset(unknown, encoding_container)