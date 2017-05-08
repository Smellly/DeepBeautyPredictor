# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2017-05-04 19:21:09.
// File name: preprocess_face_landmark.py
//
// Description:
    输入是图片，输出是68*2的facial landmark矩阵
'''

import face_landmark_detection
import numpy as np
from tqdm import tqdm

def get_path(path):
    pre_path = '/home/smelly/projects/FAC_supplement/data/Alignment_all/'
    with open(path, 'r') as f:
        namelist = f.readlines()
    pathlist = []
    for name in namelist:
        pathlist.append(pre_path + name.strip().split()[0])
    return pathlist
    
def pre_process(pathlist):
    error = []
    landmarks = []
    for img_path in tqdm(pathlist):
        try:
            flmat = face_landmark_detection.get_landmarks(img_path)
            landmarks.append(flmat)
            np.save(img_path.replace('Alignment','face_landmark').replace('.jpg',''), flmat)
        except:
            error.append(img_path+'\n')
    '''
    lm_mat = np.zeros(len(landmarks), landmarks[0].shape[0]*landmarks[0].shape[1])
    for i,j in zip(range(len(landmarks)),landmarks):
        lm_mat[i] = np.reshape(j,j.shape[0]*j.shape[1])
    # np.save(lm_mat)
    '''
    print('error in '+ str(len(error))+' images')
    with open('error.txt', 'w') as f:
        f.writelines(error)

if __name__ == '__main__':
    path = '/home/smelly/projects/FAC_supplement/data/total.txt'
    l = get_path(path)
    pre_process(l)
