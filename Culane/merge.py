import os
import numpy as np
import cv2

def Single(jpg_path):
    jpg_path = jpg_path.strip().split()[0]
    lane_path ='lane_6/'+ jpg_path.replace('.jpg','_ins.png')
    lane = cv2.imread(lane_path,-1)
    drive_path = 'drivable/'+jpg_path.replace('.jpg','_center.png')
    print(drive_path)
    drive = cv2.imread(drive_path,-1)
    drive[np.where(drive<7)] = 0
    result = drive + lane
    result[np.where(result==8)] = 1
    result[np.where(result==9)] = 2
    result[np.where(result==10)] = 3
    result[np.where(result==11)] = 4
    result[np.where(result==12)] = 5
    result[np.where(result==13)] = 6

    out_path = drive_path.replace('drivable','lane_6_drive')
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    cv2.imwrite(out_path,result)
    
def main(txt_path):
    lines = open(txt_path,'r').readlines()
    for idx,line in enumerate(lines):
        print(idx,line)
        Single(line)

if __name__ == "__main__":
    main("train.txt")
    main('val.txt')
# lane_path = "driver_23_30frame/05171102_0766.MP4/00020.jpg"
# Single(lane_path)