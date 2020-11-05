#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 下午7:07
# @Author  : Mengzhibin
# @Site    : 
# @File    : culane_instance_by_exist.py

import cv2,os
import numpy as np
from glob import glob
import shutil

class draw_lanes(object):
    def __init__(self,label_txt):
        self.root_dir = "/data/mengzb/culane/lane_6/"
        self.png_dir = "/data/mengzb/culane/lane_6_png/"
        self.img_shape = [590,1640]
        self.lanes = open(label_txt,"r").readlines()
        self.img_path = ""
        self.label_path = ""
        self.txt_path = ""
        self.label_id = []
        self.class_num = 6
        self.run()

    def run(self):
        for idx,line in enumerate(self.lanes):
            self.get_path(line)
            self.draw()
            if idx % 100 == 0:
                print(idx,line)

    def get_path(self,line):
        split_list = line.strip().split(" ")
        self.txt_path = self.root_dir + split_list[0].replace('.jpg','.lines.txt')
        self.label_path = self.txt_path.replace(".lines.txt","_ins.png")

        self.label_id = list(map(int,split_list[1:]))
        temp_id_list = []
        
        for i in range(1,self.class_num+1):
            if self.label_id[i-1] == 1:
                temp_id_list.append(i)
        self.label_id = temp_id_list

    def draw(self):
        lines = open(self.txt_path,'r').readlines()
        instance = np.zeros([self.img_shape[0], self.img_shape[1]], np.uint8)
        if len(lines) == 0:
            cv2.imwrite(self.label_path,instance)
            return

        for idx_line,line in enumerate(lines):
            axis = np.fromstring(line,dtype=float, sep=' ').reshape((-1,2))
            for line_width,i in enumerate(range(axis.shape[0]-1)):
                idx = axis.shape[0] - i -1

                if line_width > 17:
                    width = 17
                else:
                    width = line_width
                color = self.label_id[idx_line]
                cv2.line(instance,(int(axis[idx,0]),int(axis[idx,1])),(int(axis[idx-1,0]),int(axis[idx-1,1])),color=color,thickness=width+3)

        cv2.imwrite(self.label_path,instance)

def main():
    train_path = "/data/mengzb/culane/train.txt"
    # val_path = "/data/mengzb/culane/val.txt"
    draw_lanes(train_path)
    # draw_lanes(val_path)
    
if __name__ == "__main__":
    main()
    # file = "/home/dataset/CULane/driver_23_30frame/05151649_0422.MP4/00030.jpg"
    # img0 = cv2.imread(file)
    # img1 = cv2.imread("/home/dataset/CULane/laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00030.png")
    # print(np.unique(img1))

    # # print()
    # img = cv2.addWeighted(img0,1,img1*255,0.3,gamma=0)
    # cv2.imwrite('img.png',img)

