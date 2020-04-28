# -*- coding: utf-8 -*-
'''
2020.04.28 

author: tong



'''

import sys
import time
from PIL import Image, ImageDraw
# from models.tiny_yolo import TinyYoloNet
from utils.utils import *
from tool.darknet2pytorch import Darknet
import cv2



def pre_process(img_ori,resize_width,resize_height):
    width = img_ori.shape[1]
    height = img_ori.shape[0]
    img = cv2.resize(img_ori, (resize_width, resize_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return img


def draw_result(img_ori,boxes):
    width = img_ori.shape[1]
    height = img_ori.shape[0]
    for i in range(len(boxes)):
        box = boxes[i]
        print(box)
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        cls_id = box[6]
        if class_names[cls_id] == "open":
            rgb = (255,0,0)
            img_ori = cv2.putText(img_ori, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, rgb, 3)
            img_ori = cv2.rectangle(img_ori, (x1, y1), (x2, y2), rgb, 3)
        else:
            rgb = (0,255,0)
            img_ori = cv2.putText(img_ori, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, rgb, 3)
            img_ori = cv2.rectangle(img_ori, (x1, y1), (x2, y2), rgb, 3)
    return img_ori
if __name__ == '__main__':
    cfgfile = "./cfg/yolov4.cfg"
    weightfile = "./yolov4_4000.weights"
    namesfile = "./data/voc.names"
    nms_thresh = 0.4
    imgfile = "./test1.jpg"
    savename = "./predict.jpg"
    # 不使用GPU
    use_cuda = 0
    # 加载模型
    model = Darknet(cfgfile)
    model.load_weights(weightfile)
    model.eval()
    # 是否使用gpu
    if use_cuda:
        model.cuda()
    # 图片预处理
    img_ori = cv2.imread(imgfile)
    
    img = pre_process(img_ori,model.width,model.height)
    # 前向计算
    
    width = img_ori.shape[1]
    height = img_ori.shape[0]
    if use_cuda:
        img = img.cuda()
    img = torch.autograd.Variable(img)
    start_time = time.time()
    list_boxes = model(img)
    print("cost time:{}".format(time.time()-start_time))
    # nms算法筛选框
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    class_names = load_class_names(namesfile)
    boxes = nms(boxes, nms_thresh)
    result_img = draw_result(img_ori,boxes)
    print(result_img.shape)
    cv2.imwrite(savename, result_img)
