import cv2
import numpy as np
from scipy.interpolate import interp1d


def sorted_coords(y_list,x_list):
    zipped = zip(y_list,x_list)
    sort_zipped = sorted(zipped,key=lambda x:(x[0]),reverse=True)
    result = zip(*sort_zipped)
    y_list,x_list = [list(x) for x in result]
    return y_list,x_list

def interpolate(y_list,x_list,h):
    y_list,x_list = sorted_coords(y_list,x_list)
    if len(y_list)>4:
        li = interp1d(y_list,x_list,kind='cubic')
    else:
        li = interp1d(y_list,x_list)
    y_list = np.linspace(y_list[-1],y_list[0], int(y_list[0] - y_list[-1]))
    x_list = li(y_list)
    return y_list,x_list

def sort_lanes(coordinates,img_w):
    bottom_points = [c[1][-1] for c in coordinates]
    left_lanes = []
    middle_lanes = []
    right_lanes = []
    for idx in range(len(bottom_points)):
        if int(bottom_points[idx]) == 0:
            left_lanes.append(coordinates[idx])
        elif int(bottom_points[idx]) == img_w-1:
            right_lanes.append(coordinates[idx])
        else:
            middle_lanes.append(coordinates[idx])

    left_lanes = sorted(left_lanes,key=lambda x:(x[0][-1]),reverse=False)
    right_lanes = sorted(right_lanes,key=lambda x: (x[0][-1]),reverse=True)
    middle_lanes = sorted(middle_lanes,key=lambda x: (x[1][-1]),reverse=False)
    return left_lanes+middle_lanes+right_lanes

def json_xy(image_p,json_loaded):

    image = cv2.imread(image_p)
    coordinates = []
    for line in json_loaded:
        x_list = []
        y_list = []
        for c in line:
            x0 = float(c['x'])
            y0 = float(c['y'])
            if y0 not in y_list and y0<image.shape[0]:
                x_list.append(x0)
                y_list.append(y0)

        if len(x_list)<2:
            continue

        y_list,x_list = interpolate(y_list,x_list,image.shape[0])
        if len(x_list) == 0:
            continue
        coordinates.append([y_list,x_list])

    
    coordinates = sort_lanes(coordinates,image.shape[1])
    bottom_points = [c[1][-1] for c in coordinates]
    

    middle_x = int(image.shape[1]/2)
    lane_exists = np.array([0,0,0,0,0,0])
    left_index_list = np.where(np.array(bottom_points)<middle_x)[0]

    if len(left_index_list.tolist()) == 0:
        lane_exists[3:len(coordinates)+3] = 1
        return coordinates[:3],lane_exists

    middle_x_idx = left_index_list[-1]

    if middle_x_idx < 2:
        lane_exists[2-middle_x_idx:3] = 1
    else:
        lane_exists[0:3] = 1
    
    lane_exists[3:(2 + len(bottom_points)-middle_x_idx)] = 1

    return coordinates[max(middle_x_idx-2,0):middle_x_idx+4],lane_exists


def plot(image_p,coordinates,lane_exists):

    image = cv2.imread(image_p)
    instance = np.zeros([image.shape[0], image.shape[1]], np.uint8)
    color = np.where(lane_exists==1)[0][0]+1

    for line in coordinates:
        y_list,x_list = line

        h, w = image.shape[:2]
        width_mid = w/2        
        
        width_decay = (1-abs(x_list[-1] - width_mid)/width_mid)*0.6 + 0.4

        all_height = y_list[-1] - y_list[0]
        for idx in range(len(y_list)-1):
            x0 = x_list[idx]
            y0 = y_list[idx]
            x1 = x_list[idx+1]
            y1 = y_list[idx+1]
            line_width = 4 + (y1 - y_list[0])*18.0*w/1280/all_height            
            line_width = int(line_width*width_decay)
            cv2.line(instance,(int(x0),int(y0)),(int(x1),int(y1)),color=int(color),thickness=line_width)
            # cv2.line(image,(int(x0),int(y0)),(int(x1),int(y1)),color=(255,0,0),thickness=line_width)
        
        color += 1

    bn = os.path.basename(image_p)
    # cv2.imwrite('valid/{}'.format(bn),image)

    cv2.imwrite('/train/culane_format_curvelane/valid/png_label/{}'.format(bn.replace('.jpg','.png')),instance)

            
import os
import cv2
import json
# import multiprocessing
root = '/train/CurveLanes/valid/'

writer = open("val.txt","w")

for idx,f in enumerate(os.listdir(os.path.join(root,'images'))):
    
    # if idx < 6500:
    #     continue
    # f = "4c2b13c85a6ce1edb7e50fe323580801.jpg"
    print(idx,f)

    image_p = os.path.join(root,'images',f)
    json_p = os.path.join(root,'labels',f.replace('.jpg','.lines.json'))

    coordinates,lane_exists = json_xy(image_p,json.load(open(json_p))['Lines'])
    plot(image_p,coordinates,lane_exists)

    lane_exists_txt = " ".join([str(t) for t in lane_exists.tolist()])
    writer.writelines("images/{} {}\n".format(f,lane_exists_txt))

    # break

