import cv2
import json
import numpy as np

sample_img = "/root/train/images/0-001620-015-0000030.jpg"
sample_json = "/root/train/labels/0-001620-015-0000030.json"
mask_folder = "/root/train/masks"


def polygon2mask(label,img_w,img_h):
    blank = np.ones((img_w,img_h),dtype=np.uint8)
    for object_i in label['objects']:
        polygon_list = []
        for p in object_i['obj_points']:
            polygon_list.append([int(p['x']), int(p['y'])])

        cv2.fillPoly(blank, np.array([polygon_list]), 1)
    return blank

import os
json_folder = "/root/train/labels"
for json_file in os.listdir(json_folder):
    if not json_file.endswith(".json"):
        continue
    label = json.load(open(os.path.join(json_folder,json_file)))
    mask = polygon2mask(label,1200,1920)
    mask_path = os.path.join(mask_folder,json_file.replace(".json",".png"))
    cv2.imwrite(mask_path, mask)
