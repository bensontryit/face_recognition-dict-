import dlib
import numpy as np
import os
import json

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

#遍歷整個dataset
def list_jpg_files(directory):
    jpg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                jpg_files.append(os.path.join(root, file))
    return jpg_files

#從jpg檔中取得人臉的特徵向量並儲存在一個容器中
def encoding_pictures(jpg_files):
    container = np.zeros([len(jpg_files), 128])
    for i, jpg in enumerate(jpg_files):
        container[i,:] = getFeature(jpg)
    return container

#定義一個dict來儲存人臉特徵向量對應到的檔案名稱，並將特徵向量(np.array)轉為list方便儲存在json檔中
def name_dictionary(jpg_files, container):
    name_dict = {jpg: container[i].tolist() for i, jpg in enumerate(jpg_files)}  # 使用.tolist()將ndarray轉換為列表
    return name_dict

#判別此人是否存在我的dataset中
def in_dataset(unknown_pic, dict_of_data):
    unknown_encoding = getFeature(unknown_pic)
    count = 0
    for i, dataset_jpg_encoding in enumerate(dict_of_data.values()):
        dist = np.linalg.norm(unknown_encoding-dataset_jpg_encoding)
        if dist < 0.4:
            print(dist)
            print("The person is in the dataset")
            inverted_dict = {tuple(v): k for k, v in dict_of_data.items()}
            print(inverted_dict[tuple(dataset_jpg_encoding)])
            count = 1
            break
        else:
            continue
    if(count == 0):
        print("The person is not in the dataset.")

#更新json檔資料中的人臉的特徵向量
def update_data_injson(name_of_dict):
    with open('data.json', 'w') as json_file:
        json.dump(name_of_dict, json_file, indent=4)

#打開json檔取得其中的人臉特徵向量list
def open_json_data():
    with open('data.json', 'r') as json_file:
        return json.load(json_file)

#重新編碼dataset中所有人的人臉特徵
def encoding_dataset():
    dataset_directory = 'small_dataset'
    jpg_files = list_jpg_files(dataset_directory)
    encoding_container = encoding_pictures(jpg_files)
    name_dict = name_dictionary(jpg_files, encoding_container)
    update_data_injson(name_dict)

while(True):
    keyboard = input("Do you want to encoding the whole image dataset?[Yes/No] ")
    if(keyboard == "Yes"):
        encoding_dataset()
        keyboard_1 = input("Do you want to do the face recognition?[Yes/No] ")
        if(keyboard_1 == "Yes"):
            unknown = input("please enter the unknown picture file name ")
            load_data = open_json_data()
            in_dataset(unknown, load_data)
            break
        elif(keyboard_1 == "No"):
            print("encoding the whole picture successfully.")
            break
    elif(keyboard == "No"):
        recog = input("Do you want to do the face recognition?[Yes/No] ")
        if(recog == "Yes"):
            unknown = input("please enter the unknown picture file name ")
            load_data = open_json_data()
            in_dataset(unknown, load_data)
            break
        elif(recog == "No"):
            break
        else:
            print("input error.")
    else:
        print("input error.")