import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

class AddLane(object):
    def __init__(self):
        self.img_shape = [590, 1640]
        self.ipm_shape = [600,300]
        self.M = {'driver_23_30frame': np.array([[-1.53600965e+00 ,-1.07981516e+00 , 2.92863875e+02],
                                                [-2.07323223e+00 ,-4.09225080e+00 , 6.73187551e+02],
                                                [-3.62980677e-03, -7.34283219e-03 , 1.00000000e+00]]),
            'driver_161_90frame': np.array([[-5.65937626e-01 ,-5.74099150e-01 , 2.19251126e+02],
                                                [ 5.33101487e-02 ,-2.53106967e+00 , 6.80726347e+02],
                                                [ 2.66550897e-04 ,-4.20885169e-03 , 1.00000000e+00]]),
            'driver_182_30frame': np.array([[-2.32985033e+00 ,-2.06499200e+00  ,4.14341766e+02],
                                                [-1.66128026e+00, -7.99766711e+00 , 8.21367849e+02],
                                                [-2.35131626e-03,-1.50740244e-02 , 1.00000000e+00]])}

    def ipm_image(self,image_path,image_folder):
        image = cv2.imread(image_path,-1)
        image = image[240:,:,:]
        resize_image = cv2.resize(image, (300, 600),interpolation=cv2.INTER_NEAREST)

        ipm_image = cv2.warpPerspective(resize_image, self.M[image_folder], (300, 600),flags=cv2.INTER_NEAREST)
        return ipm_image

    def ipm_label(self,label_path,image_folder):
        image = cv2.imread(label_path,-1)
        image = image[240:, :]
        image[np.where(image<7)] = 0
        resize_image = cv2.resize(image, (300, 600),interpolation=cv2.INTER_NEAREST)

        ipm_image = cv2.warpPerspective(resize_image, self.M[image_folder], (300, 600),flags=cv2.INTER_NEAREST)
        return ipm_image

    def ipm_points(self,axis,image_folder):
        axis[:,1] = axis[:,1] - 240
        ratio_x = 300 / 1640
        ratio_y = 600 / 350
        axis[:,1] = axis[:,1] * ratio_y
        axis[:,0] = axis[:,0] * ratio_x
        axis = np.insert(axis,2,1,1)
        axis = np.transpose(axis)

        Perspective = np.dot(self.M[image_folder], axis)
        X_Y = Perspective[:2, :]
        scale = Perspective[2:, :]
        Perspective_points = X_Y / scale
        return Perspective_points.astype(int)

    def back_points(self,points,image_folder):
        points = points.copy()
        points = np.insert(points,2,1,0)
        M_inv = np.linalg.inv(self.M[image_folder])
        image_points = np.dot(M_inv,points)
        X_Y = image_points[:2, :]
        scale = image_points[2:, :]
        Perspective_pts = X_Y / scale

        ratio_x = 1640 / 300
        ratio_y = 350 / 600
        Perspective_pts[1,:] = Perspective_pts[1,:] * ratio_y + 240
        Perspective_pts[0,:] = Perspective_pts[0,:] * ratio_x

        return Perspective_pts.astype(int)

    def judge_left(self,label,line):
        # polygon = [(50,0),(40,99),(99,99)]
        polygon = []
        polygon.append((0,0))
        for i in range(line.shape[1]):
            polygon.append((line[0,i],line[1,i]))
        polygon.append((0,600))
        
        img = Image.new('L', (self.ipm_shape[1], self.ipm_shape[0]), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=255)
        mask = np.array(img)

        diff = mask - label
        area = len(np.where(diff == (255-7))[0])
        diff[np.where(diff != (255-7))] = 0
        if area > 1000:
            return True
        else:
            return False
    
    def judge_right(self,label,line):
        polygon = []
        polygon.append((300,0))
        for i in range(line.shape[1]):
            polygon.append((line[0,i],line[1,i]))
        polygon.append((300,600))
        
        img = Image.new('L', (self.ipm_shape[1], self.ipm_shape[0]), 0)
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=255)
        mask = np.array(img)
        
        diff = mask - label
        area = len(np.where(diff == (255-7))[0])
        diff[np.where(diff != (255-7))] = 0
        if area > 1000:
            return True
        else:
            return False

    def draw_points(self,image,pts):
        # image = cv2.resize(image,(300,600))
        for i in range(pts.shape[1]-1):
            cv2.line(image,(pts[0,i],pts[1,i]),(pts[0,i+1],pts[1,i+1]),255,4)
        return image

    def cal_width(self,line_0,line_1):
        w0 = np.mean(line_0[0,:],axis=0)
        w1 = np.mean(line_1[0,:],axis=0)
        return int(w1 - w0)

    def np2string(self,line):
        line = np.transpose(line).flatten()
        line = [str(i) for i in line.tolist()]
        line_str = " ".join(line)
        return line_str

    def save_txt(self,lines,input_txt_path):
        save_path = input_txt_path.replace("lane","lane_6")
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        writer = open(save_path,'w')
        for line in lines:
            writer.writelines(line.strip()+'\n')

    def resolve_txt(self,line_idx):
        line_idx = line_idx.strip().strip('/')
        split_r = line_idx.split()
        exists = split_r[-4:]
        label_path = 'drivable/'+split_r[0].replace(".jpg","_center.png")
        image_folder = label_path.split('/')[-3]
        txt_path = 'lane/' + split_r[0].replace(".jpg",".lines.txt")
        lines = open(txt_path,'r').readlines()

        ipm_image = self.ipm_label(label_path,image_folder)

        if len(lines) < 3:
            exists.insert(0,'0')
            exists.append('0')
            self.save_txt(lines,txt_path)
            return split_r[0] + " "+" ".join(exists)

        lanes = []
        for idx_line,line in enumerate(lines):
            axis = np.fromstring(line,dtype=float, sep=' ').reshape((-1,2))

            Perspective_points = self.ipm_points(axis,image_folder)
            temp = np.zeros_like(Perspective_points)

            for idx,i in enumerate(np.argsort(Perspective_points)[1]):
                temp[:,idx] = Perspective_points[:,i]
            
            lanes.append(temp)

        if exists[0] == '0':
            w = self.cal_width(lanes[0],lanes[1])
        else:
            w = self.cal_width(lanes[1],lanes[2])

        left_lane = None 
        if exists[0] == '1':
            if self.judge_left(ipm_image,lanes[0]):
                new_lane = lanes[0].copy()
                new_lane[0,:] = new_lane[0,:] - w
                left_lane = self.back_points(new_lane,image_folder)
                # image = self.draw_points(image,left_lane)
                # lanes.insert(0,new_lane)

        right_lane = None
        if exists[3] == '1':
            if self.judge_right(ipm_image,lanes[-1]):
                new_lane = lanes[-1].copy()
                new_lane[0,:] = new_lane[0,:] + w
                right_lane = self.back_points(new_lane,image_folder)
                # image = self.draw_points(image,right_lane)

        exists.insert(0,'0')
        exists.append('0')
        if left_lane is not None:
            left_lane = np.flip(left_lane,1)
            left = self.np2string(left_lane)
            lines.insert(0,left)
            exists[0] = '1'

        if right_lane is not None:
            right_lane = np.flip(right_lane,1)
            right = self.np2string(right_lane)
            lines.append(right)
            exists[-1] = '1'
        self.save_txt(lines,txt_path)
        return split_r[0] + " "+" ".join(exists)
    def exception_process(self,line_idx):
        line_idx = line_idx.strip().strip('/')
        split_r = line_idx.split()
        exists = split_r[-4:]
        txt_path = 'lane/' + split_r[0].replace(".jpg",".lines.txt")
        lines = open(txt_path,'r').readlines()
        self.save_txt(lines,txt_path)
        exists.insert(0,'0')
        exists.append('0')
        return split_r[0] + " "+" ".join(exists)

# txt_path = "/driver_23_30frame/05151649_0422.MP4/02220.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/02220.png 0 1 1 1"
# txt_path = "/driver_23_30frame/05151643_0420.MP4/00240.jpg abc 1 1 1 0"
# # add right
txt_path = "/driver_23_30frame/05160717_0443.MP4/01095.jpg abc 0 1 1 1"
txt_path = "/driver_23_30frame/05170838_0718.MP4/00650.jpg abc 1 1 1 1"
lines = open("list/train_gt.txt",'r').readlines()
Add = AddLane()
writer_gt = open("train.txt",'w')
# exists = Add.resolve_txt(txt_path)
# import sys 
# sys.exit(0)
idx = 0
for txt_path in lines:
    try:
        exists = Add.resolve_txt(txt_path)
        writer_gt.writelines(exists.strip()+'\n')
    except Exception as e:
        exists = Add.exception_process(txt_path)
        writer_gt.writelines(exists.strip()+'\n')
    idx += 1
    if idx % 100:
        print(idx,txt_path)