# coding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np

class FaceRecognition(object):
    def __init__(self):
        super(FaceRecognition, self).__init__()
        # 定义检测器
        self.detector = dlib.get_frontal_face_detector()
        # 输出人脸图像的大小
        self.img_size = 150
        # 记载数据集
        # 关键点预测
        self.predictor = dlib.shape_predictor(r'./shape_predictor_68_face_landmarks.dat')
        # 人脸识别
        self.recognition = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    def get_face(self, image):
        # image是数组
        # 人脸检测
        dets = self.detector(image, 1)
        if len(dets) == 1:
            # faces = dlib.full_object_detections()
            # 关键点提取
            shape = self.predictor(image, dets[0])
            # print("Computing descriptor on aligned image ..")
            # 人脸对齐 face alignment
            images = dlib.get_face_chip(image, shape, size=self.img_size)

            # 计算对齐后人脸的128维特征向量
            face_descriptor_from_prealigned_image = self.recognition.compute_face_descriptor(images)
            face_features = np.array(face_descriptor_from_prealigned_image)
            return images, face_features
        else:
            return None

if __name__=='__main__':
    img_1 = r'./human_face/an1.jpg'
    img_1 = cv2.cvtColor(cv2.imread(img_1),cv2.COLOR_BGR2RGB)
    detection_recognition = FaceRecognition()
    face,points = detection_recognition.get_face(img_1)
    plt.imshow(face)
    plt.show()
    print(points)

