Readme

这个项目是 ACM MM 15年期刊 Deep Face Beautification 中 Deep Beauty Predictor 用 TensorFlow 复现的版本

首先用 
```
python preprocess_face_landmark.py
````
 处理照片，得到 Facial Landmark 的68*2的矩阵
然后
```
python NeuralNetwork4ACM.py
```
执行即可

注意修改相应的路径