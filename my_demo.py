import os
import sys
sys.path.append("..")
sys.path.append("./process")
sys.path.append("./model")

import torch
import cv2
import numpy as np
from Detect_humanface import FaceRecognition
import matplotlib.pyplot as plt

def get_test_model(model_name, num_class = 2, modality = 'color',pretrained_file = None):
    if model_name == 'baseline':
        from model_fusion.model_baseline_SEFusion import FusionNet
    elif model_name == 'model_A':
        from model_fusion.FaceBagNet_model_A_SEFusion import FusionNet
    elif model_name == 'model_B':
        from model_fusion.FaceBagNet_model_B_SEFusion import FusionNet
    elif model_name == 'model_C':
        from model_fusion.FaceBagNet_model_C_SEFusion import FusionNet
    elif model_name == 'resnet18':
        from model_fusion.model_baseline_Fusion import FusionNet

    net = FusionNet(num_class=num_class, modality=modality)
    # net = torch.nn.DataParallel(net)
    # print(net)
    net.load_state_dict(torch.load(pretrained_file))
    # net = net.cuda()
    net.eval()
    return net

def predict(model,img):
    logit,_,_ = model(img)
    print(logit)
    label = np.argmax(logit.cpu().detach().numpy())
    return label

def get_img_tensor(img,size):
    img = cv2.resize(img,size)
    img = np.transpose(img, (2, 0, 1))
    img = img/255.0
    img_tensor = torch.FloatTensor(img).view(1, 3, size[0], size[1])
    # img_tensor = img_tensor.cuda()
    return img_tensor

def test_image_list(img_root,model):
    # 以下代码用于测试文件夹中全是图片的情况
    plt.ion()
    img_list = os.listdir(img_root)
    for path in img_list:
        img_path = os.path.join(img_root, path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        face = FaceRecognition().get_face(img)
        if face:
            face_img = face[0]
            img_tensor = get_img_tensor(face_img,(112,112))
            result = predict(model, img_tensor)
            plt.imshow(face_img)
            if result == 1:
                plt.title('%s is a real face' % path)
            else:
                plt.title('%s is a fake face' % path)
        else:
            plt.imshow(cv2.resize(img,(112,112)))
            plt.title('%s has no face' % path)
        plt.pause(0.1)
    plt.ioff()
    plt.show()

def test_video(video_path, model):
    # 以下代码用于测试视频
    plt.ion()
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    img_number = 0
    if not success:
        print('Failed to read video')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = FaceRecognition().get_face(img)
        if face:
            face_img = face[0]
            img_tensor = get_img_tensor(face_img, (112,112))
            result = predict(model, img_tensor)
            plt.imshow(img)
            if result == 1:
                plt.title('this picture is a real face ,frame number = %s' % img_number)
            else:
                plt.title('this picture is a fake face ,frame number = %s' % img_number)
        else:
            plt.imshow(img)
            plt.title('this picture has no face ,frame number = %s' % img_number)
        img_number += 1
        plt.pause(0.02)
    plt.ioff()
    plt.show()

def recognize_face(rec_point, dst_point, threshold = 0.5):
    distance = np.sqrt(np.sum((rec_point - dst_point) * (rec_point - dst_point)))
    if distance < threshold:
        print('same people')
    else:
        print('different people')


if __name__=='__main__':
    model_name = 'baseline'
    pre_file = './models/'+model_name+'_color_64/checkpoint/global_min_acer_model.pth'
    modality = 'color'
    model = get_test_model(model_name, modality=modality, pretrained_file=pre_file)
    # img_root = './test_image'
    # test_image_list(img_root, model)
    video_path = './b.mp4'
    test_video(video_path, model)

