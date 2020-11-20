

json_path = "/home/mengzhibin/Ti/mmdetection/data/ark/fangzhou_BJ_ddd_traffic_sign_label_coco_format.json"

import json


with open(json_path,'r') as load_f:
    load_dict = json.load(load_f)

print(load_dict.keys())

train_dict = dict()
train_dict['type'] = load_dict['type']
train_dict['categories'] = load_dict['categories']
train_dict['images'] = []


val_dict = dict()
val_dict['type'] = load_dict['type']
val_dict['categories'] = load_dict['categories']
val_dict['images'] = []

print(len(load_dict['images']),len(load_dict['annotations']))

def get_annotation(annotations_list,image_list):
    ann_list = []
    image_id_list = []
    for image in image_list:
        image_id_list.append(image['id'])

    for ann in annotations_list:
        if ann['image_id'] in image_id_list:
            ann_list.append(ann)

    return ann_list

for idx in range(len(load_dict['images'])):
    image_id = load_dict['images'][idx]['id']

    if idx<5000:
        val_dict['images'].append(load_dict['images'][idx])
    else:
        train_dict['images'].append(load_dict['images'][idx])

val_dict['annotations'] = get_annotation(load_dict['annotations'],val_dict['images'])
print("val_dict finish")
train_dict['annotations'] = get_annotation(load_dict['annotations'],train_dict['images'])
print("train_dict finish")

with open("train.json","w") as dump_f:
    json.dump(train_dict,dump_f,indent=4)

with open("val.json","w") as dump_f:
    json.dump(val_dict,dump_f,indent=4)
