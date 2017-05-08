# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2017-05-08 19:05:50.
// File name: face_landmark_detection.py
//
// Description:
    输入是图片，输出是68*2的facial landmark矩阵
'''

import dlib
import numpy
from skimage import io

def get_landmarks(faces_path):
    # faces_path = '/home/smelly/projects/FAC_supplement/data/Alignment_all/Origin_jaffe@YM.SA1.55.jpg'
    predictor_path = "../models/shape_predictor_68_face_landmarks.dat"
    #与人脸检测相同，使用dlib自带的frontal_face_detector作为人脸检测器
    detector = dlib.get_frontal_face_detector()
    #使用官方提供的模型构建特征提取器
    predictor = dlib.shape_predictor(predictor_path)
    #使用skimage的io读取图片
    img = io.imread(faces_path)
    rects = detector(img, 1)
    return numpy.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])

#多张脸使用的一个例子
def get_landmarks_m(im):
    dets = detector(im, 1)
    #脸的个数
    print("Number of faces detected: {}".format(len(dets)))
    for i in range(len(dets)):
        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])
        for i in range(68):
            #标记点
            im[facepoint[i][1]][facepoint[i][0]] = [232,28,8]        
    return im    

# -------------test-----------------
'''
#打印关键点矩阵
print("face_landmark:")

print(get_landmarks(img))

#等待点击
# dlib.hit_enter_to_continue()
'''
