#!/usr/bin/env python3

import torch
import numpy as np
import cv2

import sys, os
sys.path.append('/home/sim/simulator/')
sys.path.append('/home/sim/simulator/yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, xyxy2xywh

from PIL import Image

import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import random
from PIL import Image, ImageDraw, ImageFont

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class YoloWrapper:
    def __init__(self, model_param_path, device = DEVICE):
        self.model = attempt_load(model_param_path)
        self.model.to(device)
        print('model loadded using the file at ', model_param_path)
        self.color = color_list()

    def get_predictions(self, torch_images):
        print('torch_images size:', torch_images.size())
        result = self.model(torch_images)
        pred = non_max_suppression(result[0])   # Apply NMS
        return pred

    def draw_image_w_predictions(self, torch_images, show=False, wait=1000):
        np_images = torch_images.permute(0,2,3,1).cpu().numpy()
        np_images_uint8 = np.clip(np_images*255, 0, 255).astype(np.uint8)
        torch_images = torch_images.detach()
        pred = self.get_predictions(torch_images)
        for i in range(len(pred)):
            detection = pred[i]
            cv2_images_uint8 = np_images_uint8[i].copy()
            np_pred = []
            if len(detection)>0:
                for box in detection:
                    box = box.cpu().numpy()
                    xyxy = box[:4].astype(int)
                    class_int =  int(box[-1])
                    plot_one_box(xyxy, cv2_images_uint8, color=self.color[class_int%10], label=self.model.names[class_int])
                    np_pred.append(box)
            if show:    
                cv2.imshow('prediction'+str(i), cv2_images_uint8)
                cv2.waitKey(wait)
        return cv2_images_uint8, np.array(np_pred)

def example_use_yolo():
    cwd = os.getcwd()
    print(cwd)

    yolo_model = YoloWrapper('yolo_param/yolov5s.pt')

    ### Load the single image ###
    data = np.asarray(Image.open('yolo_sample_images/400.png').convert('RGB'))
    x_image = torch.FloatTensor(data).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
    print('x_image', x_image.size(), 'min:', x_image.min(), 'max:', x_image.max())

    ### Use the model ### 
    pred = yolo_model.get_predictions(x_image)

    print(pred)

    ### Use the model to draw prediction ### 
    cv2_images, np_pred = yolo_model.draw_image_w_predictions(x_image, False)

    # print(type(yolo_model.model))
    # print(dir(yolo_model.model))


if __name__ == "__main__":

    example_use_yolo()
