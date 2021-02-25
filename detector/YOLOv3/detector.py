import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from utils.util import *

from detector.YOLOv3.darknet import Darknet
import yaml
import sys

def get_detector_configuration(config_file):
    ## Open config file 
    with open(config_file) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
            sys.exit()

    cfgfile = config['CFG']
    weightsfile = config['WEIGHT']
    data = config['CLASS_NAMES']
    confidence = config['SCORE_THRESH']
    nms_thesh = config['NMS_THRESH']
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model.net_info["height"] = 160
    inp_dim = int(model.net_info["height"])

    CUDA = torch.cuda.is_available()
    print("CUDA is: {}".format(CUDA))
    if CUDA:
        model.cuda()
    model.eval()

    return model, data, confidence, nms_thesh, inp_dim, CUDA

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, classes, your_class):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    if label in your_class:
        color = (0,255,0)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
        return img

def inp_to_image(inp):
    inp = inp.cpu().squeeze()
    inp = inp*255
    try:
        inp = inp.data.numpy()
    except RuntimeError:
        inp = inp.numpy()
    inp = inp.transpose(1,2,0)

    inp = inp[:,:,::-1]
    return inp

def yolo_output(frame, model, data_class, your_class, confidence, nms_thesh, CUDA, inp_dim):
    """
    Get the labeled image and the bounding box coordinates.
    """
    num_classes = 80
    bbox_attrs = 5 + num_classes
    img, orig_im, dim = prep_image(frame, inp_dim)

    im_dim = torch.FloatTensor(dim).repeat(1,2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

    output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
#            im_dim = im_dim.repeat(output.size(0), 1)
    output[:,[1,3]] *= frame.shape[1]
    output[:,[2,4]] *= frame.shape[0]

    classes = load_classes(data_class)
    box = list([])
    list(map(lambda x: write(x, orig_im, classes, your_class), output))
    for i in range(output.shape[0]):
        if int(output[i, -1]) == 0:
            c1 = tuple(output[i,1:3].int())
            c2 = tuple(output[i,3:5].int())
            box.append([c1[0].item(),c1[1].item(), c2[0].item(),c2[1].item()])

    return orig_im, box
