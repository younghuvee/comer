import numpy as np
import sys
import os
import cv2
import random
import json


def trace_transform(txt_path):
    trace = []
    with open(txt_path, "r", encoding='utf-8') as f:
        # 一体机中true和false替换为True和False
        line = f.readlines()[0].replace('false', 'False')
        line = line.replace('true', 'True')
        dic_json = eval(line)

    points = dic_json['points']
    if len(points) == 0:
        os.remove(txt_path)
        print('remove: ', txt_path)

    for ii in range(len(points)):
        for jj in range(len(points[ii])):
            point = []
            point.append(int(points[ii][jj]['x']))
            point.append(int(points[ii][jj]['y']))
            trace.append(point)
        trace.append([-1, -1])

    return trace

def trace_transform_json(points):
    trace = []
    for ii in range(len(points)):
        for jj in range(len(points[ii])):
            point = []
            point.append(int(points[ii][jj]['x']))
            point.append(int(points[ii][jj]['y']))
            trace.append(point)
        trace.append([-1, -1])

    return trace

def trace2image(trace_0):
    # 映射
    padding = 25
    trace_x = []
    trace_y = []
    for i in range(len(trace_0)):
        if trace_0[i] != [-1, -1]:
            trace_x.append(trace_0[i][0])
            trace_y.append(trace_0[i][1])
    image_h = max(trace_y) - min(trace_y)
    image_w = max(trace_x) - min(trace_x)
    padding_image_h = image_h + padding * 2
    padding_image_w = image_w + padding * 2

    # ratio
    input_h = 200
    input_w = 800
    ratio_h = padding_image_h / input_h
    ratio_w = padding_image_w / input_w
    ratio = max(ratio_w, ratio_h)
    
    trace = []
    for i in range(len(trace_0)):
        if trace_0[i] == [-1, -1]:
            trace.append([-1, -1])
            continue
        point = []
        point.append(int(trace_0[i][0] / ratio))
        point.append(int(trace_0[i][1] / ratio))
        trace.append(point)

    trace_x = []
    trace_y = []
    for i in range(len(trace)):
        if trace[i] != [-1, -1]:
            trace_x.append(trace[i][0])
            trace_y.append(trace[i][1])
    image_h = max(trace_y) - min(trace_y)
    image_w = max(trace_x) - min(trace_x)
    pad = int(padding / ratio)
    I = np.zeros((image_h + pad * 2, image_w + pad * 2), dtype=np.uint8)

    for row in range(image_h + pad * 2):
        for col in range(image_w + pad * 2):
            I[row, col] = 0

    image = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR)

    num = 0
    for i in range(len(trace)):
        if num == 0:
            list_trace = []
        if trace[i] != [-1, -1]:
            if len(list_trace) > 0:
                cv2.line(image, (list_trace[-1][0] - min(trace_x) + pad, list_trace[-1][1] - min(trace_y) + pad),
                         (trace[i][0] - min(trace_x) + pad, trace[i][1] - min(trace_y) + pad), (255, 255, 255), 2)
            list_trace.append(trace[i])
            num += 1
        else:
            num = 0
    image_h = image.shape[0]
    image_w = image.shape[1]
    top = 0
    bottom = input_h - image_h
    left = 0
    right = input_w - image_w
    image_resize = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return cv2.cvtColor(image_resize, cv2.COLOR_BGR2GRAY)
