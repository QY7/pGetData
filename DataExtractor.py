
import cv2
# from cv2 import cvtColor,imshow,threshold,erode,getStructuringElement,MORPH_RECT,THRESH_OTSU,THRESH_BINARY_INV,COLOR_BGR2GRAY
from numpy import array,median,argwhere,power,tile,newaxis,savetxt,uint8
from enum import Enum
from math import log10
import numpy as np
import keyboard
# 18, 121, 190
_red_lim = [10,30]
_green_lim = [100,150]
_blue_lim = [180,220]

class AxisType(Enum):
    LINEAR = 1
    LOG = 2

class DataExtractor:
    def __init__(self):
        self.lower_color_lim = [0,0,0]
        self.higher_color_lim = [255,255,255]

        self._fuzzy_range = 50 #fuzzy range refers to picking range(0~255)
        self._step = 10
        self._blend_transparency = 0.3
        self.grid_size = 3
        self.image_width = 0
        self.image_height = 0
        self.startX = 0.5 #x坐标最小
        self.startY = 1 #x坐标最大
        self.endX =100 #y坐标最小
        self.endY = 4000 #y坐标最大

        self.xaxis_type = AxisType.LINEAR
        self.yaxis_type = AxisType.LINEAR

        self.original_img  = 0
        self.color_set = None
        self.erase_range = 5
        self.drawing_mode = False
        
    def set_fuzzy_color_range(self,pixdata):
        color = cv2.cvtColor(uint8([[pixdata]]),cv2.COLOR_BGR2HSV)
        # print(color)
        self.lower_color_lim = array([color[0][0][0],20,20])
        self.higher_color_lim = array([color[0][0][0],255,255])

    def get_color_mask(self,img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.lower_color_lim,self.higher_color_lim)
        return mask
    def set_axis_range(self,custom_range):
        # x1 x2 y1 y2
        [self.startX,self.endX,self.startY,self.endY] = custom_range
        
    def rm_grid_bywidth(self):
        gray = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)#读取灰度
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Remove grid
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.grid_size,self.grid_size))
        return cv2.erode(thresh,horizontal_kernel)
        # return horizontal_kernel

    def rm_grid(self,grid_width,grid_height):
        # 灰度图
        gray = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        # 得到原图的拷贝，避免污染原图
        # 二值化
        # 由于有的网格的颜色灰度比较浅，非常接近白色的255，需要把阈值取得比较高，让尽可能多的点认定为网格点
        ret, binary = cv2.threshold(gray, 250, 255, 0)
        inv = 255 - binary
        horizontal_img = inv
        vertical_img = inv
        # 动态调节Length，可以以图像的长和宽为参考
        # 删除竖向的线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (grid_height,1))
        horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
        horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)
        
        # 删除横向的线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,grid_width))
        vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
        vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)
        # 把横向和竖向加起来
        mask_img = horizontal_img + vertical_img
        mask_img_inv = cv2.bitwise_not(mask_img)

        result = cv2.bitwise_and(inv,mask_img_inv)

        return result

    def extract_data(self,data):
        result = []
        # data = data.T

        for i in range(0,self.image_width,self._step):
            # 如果有黑色像素，取中值
            # 注意行列和矩阵的维度的对应
            # 矩阵中，第一维为行，第二为列。所以对x坐标进行循环， 就应该取出对应的列的所有行
            if(255 in data[:,i]):
                ldx = self.image_height-median(argwhere(data[:,i]==255))
                result.append([i,ldx])

        # print(array(result))
        return array(result)

    def data_mapping(self,data):
        if(self.xaxis_type == AxisType.LINEAR):
            data[:,0] = data[:,0]/self.image_width*(self.endX-self.startX)+self.startX
        else:
            data[:,0] = power(10,(data[:,0]/self.image_width*(log10(self.endX)-log10(self.startX))+log10(self.startX)))
        if(self.yaxis_type == AxisType.LINEAR):
            data[:,1] = data[:,1]/self.image_height*(self.endY-self.startY)+self.startY
        else:
            data[:,1] = power(10,(data[:,1]/self.image_height*(log10(self.endY)-log10(self.startY))+log10(self.startY)))

        return data

    def erase_noise(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_mode = True

        if event == cv2.EVENT_MOUSEMOVE and self.drawing_mode:
            self.mask[y-self.erase_range:y+self.erase_range,x-self.erase_range:x+self.erase_range]=0
            masked_img = tile(self.mask[:,:,newaxis],3)
            blender=cv2.addWeighted(self.original_img,self._blend_transparency,masked_img,1,0)
            cv2.imshow("masked image",blender)
        
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing_mode = False


    def pick_color(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.color_set = self.original_img[y][x]
            print("当前选择颜色为:")
            print(self.color_set)

    def filter_grid(self):
        grid_mask = self.rm_grid(int(self.image_width*0.8),int(self.image_height*0.8))
        # 扩展数组
        grid_mask_new = np.zeros([*grid_mask.shape,3],dtype=np.uint8)
        # 填充颜色
        grid_mask_new[grid_mask[:,:]==255,1]=255
        blender=cv2.addWeighted(self.original_img,1,grid_mask_new,1,0)
        cv2.imshow("original image",blender)
        self.mask = grid_mask_new[:,:,1]

    def extract(self,color_mode):
        if(color_mode == False):
            self.mask = self.rm_grid(int(0.4*self.image_width),int(0.4*self.image_height))

        else:
            # 显示原图，让用户选择颜色
            # 通过一个函数触发修改grid大小的函数
            cv2.namedWindow("original image")
            cv2.moveWindow("original image", 40,30)  # Move it to (40,30)
            cv2.imshow("original image",self.original_img)
            cv2.setMouseCallback("original image",self.pick_color)

            # 检测键盘输入，如果键盘输入回车，进入下一步
            # 筛选颜色
            while(1):
                k = cv2.waitKey(20) & 0xFF
                if k == 27:
                    break
                elif k == 13:
                    # 直到检测到颜色输入为止
                    cv2.destroyWindow("original image")
                    break
            self.set_fuzzy_color_range(self.color_set)
            self.mask = self.get_color_mask(self.original_img)

        # keyboard.wait('a')
        cv2.destroyWindow("original image")
        self.filtered_img = tile(self.mask[:,:,newaxis],3)

        blender=cv2.addWeighted(self.original_img,self._blend_transparency,self.filtered_img,1,0)
        
        cv2.namedWindow("masked image")
        cv2.moveWindow("masked image", 40,30)  # Move it to (40,30)
        cv2.imshow("masked image",blender)

        # # 让用户去除不要的数据点
        cv2.setMouseCallback('masked image',self.erase_noise)
        while(1):
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

            elif k == 13:
                # cv2.imshow("after",self.binary_data_without_border)
                cv2.destroyWindow("masked image")
                extracted_data = self.extract_data(self.mask)
                # 数据点映射坐标
                mapped_data = self.data_mapping(extracted_data)
                return mapped_data