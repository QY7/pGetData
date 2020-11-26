from PIL import Image
from PIL import ImageGrab
import cv2
import numpy
from enum import Enum
import matplotlib.pyplot as plt
from math import log10
# 18, 121, 190
_red_lim = [10,30]
_green_lim = [100,150]
_blue_lim = [180,220]

class AxisType(Enum):
    LINEAR = 1
    LOG = 2

class DataExtractor:
    def __init__(self):
        self._red_lim = [0,20]
        self._green_lim = [0,20]
        self._blue_lim = [0,20]
        self._fuzzy_range = 50 #fuzzy range refers to picking range(0~255)
        self._step = 10
        # self._x1 = 0
        # self._x2 = 580
        # self._y1 = 13
        # self._y2 = 430
        self._filter_size = 3
        self.width = 0
        self.height = 0
        self.startX = 0.5#从datasheet读取的数据
        self.startY = 1 # 从datasheet读取的数据
        self.endX =100
        self.endY = 4000

        self.xaxis_type = AxisType.LINEAR
        self.yaxis_type = AxisType.LOG

    def set_fuzzy_color_range(self,pixdata):
        self._red_lim[0] = pixdata[0] - self._fuzzy_range if pixdata[0]>self._fuzzy_range else 0
        self._red_lim[1] = pixdata[0] + self._fuzzy_range if pixdata[0]<255-self._fuzzy_range else 255
        self._green_lim[0] = pixdata[1] - self._fuzzy_range if pixdata[1]>self._fuzzy_range else 0
        self._green_lim[1] = pixdata[1] + self._fuzzy_range if pixdata[1]<255-self._fuzzy_range else 255
        self._blue_lim[0] = pixdata[2] - self._fuzzy_range if pixdata[2]>self._fuzzy_range else 0
        self._blue_lim[1] = pixdata[2] + self._fuzzy_range if pixdata[2]<255-self._fuzzy_range else 255

    # 筛选颜色范围
    def check_valid(self,pixdata):
        if(pixdata[0]<self._red_lim[0] or pixdata[0]>self._red_lim[1]):
            return False
        if(pixdata[1]<self._green_lim[0] or pixdata[1]>self._green_lim[1]):
            return False
        if(pixdata[2]<self._blue_lim[0] or pixdata[2]>self._blue_lim[1]):
            return False
        return True

    def filter_color_range(self,img):
        original_image = numpy.array(img)#转换为cv2能用的格式
        self.image_width = img.size[0]
        self.image_height = img.size[1]
        pixdata = img.load()
        # cv2.imshow("original",original_image)

        for y in range(img.size[1]):
            for x in range(img.size[0]):
                # 二值化
                if not self.check_valid(pixdata[x,y]):
                    pixdata[x, y] = (255,255,255)
                else:
                    pixdata[x, y] = (0,0,0)
                    
        image = numpy.array(img)#转换为cv2能用的格式
        image = image[:, :, :].copy()
        # cv2.imshow("color_filter",image)
        
        # result是一个y*x*color_channel的数组
        result_wo_grid = self.remove_grid(img)

        return result_wo_grid

    def set_axis_type(self,xtype = AxisType.LINEAR,ytype = AxisType.LINEAR):
        self.xaxis_type = xtype
        self.yaxis_type = ytype

    def set_axis_range(self,custom_range):
        # x1 x2 y1 y2
        [self.startX,self.endX,self.startY,self.endY] = custom_range
        
    def remove_grid(self,pixdata):
        image = numpy.array(pixdata)
        image = image[:, :, :].copy()  
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#读取灰度
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove grid
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._filter_size,self._filter_size))
        result = cv2.erode(thresh,horizontal_kernel)
        return result

    def extract(self,img):
        # self.width = img.size[0]
        # self.height = img.size[1]
        # 根据颜色范围选择要用到的数据
        binary_data = self.filter_color_range(img)
        # 去除边界
        binary_data_without_border = binary_data
        # binary_data_without_border = self.remove_boarder(binary_data,[self._x1,self._x2,self._y1,self._y2])
        # 提取数据点
        extracted_data = self.extract_data(binary_data_without_border)
        # 绘制数据点
        self.plot_value(extracted_data)

    def remove_boarder(self,data,filter_range):
        # filterrange，四元向量，分别是x1,x2,y1,y2
        return data[filter_range[2]:filter_range[3],filter_range[0]:filter_range[1]]
        
    def extract_data(self,data):
        result = []
        # data = data.T

        for i in range(0,self.image_width,self._step):
            # 如果有黑色像素，取中值
            # 注意行列和矩阵的维度的对应
            # 矩阵中，第一维为行，第二为列。所以对x坐标进行循环， 就应该取出对应的列的所有行
            if(255 in data[:,i]):
                ldx = self.image_height-numpy.median(numpy.argwhere(data[:,i]==255))
                result.append([i,ldx])

        # print(numpy.array(result))
        return numpy.array(result)

    def data_mapping(self,data,axis_type = AxisType.LINEAR):
        if(self.xaxis_type == AxisType.LINEAR):
            data[:,0] = data[:,0]/self.image_width*(self.endX-self.startX)+self.startX
        else:
            data[:,0] = numpy.power(10,(data[:,0]/self.image_width*(log10(self.endX)-log10(self.startX))+log10(self.startX)))
        if(self.yaxis_type == AxisType.LINEAR):
            data[:,1] = data[:,1]/self.image_height*(self.endY-self.startY)+self.startY
        else:
            data[:,1] = numpy.power(10,(data[:,1]/self.image_height*(log10(self.endY)-log10(self.startY))+log10(self.startY)))

        return data

    def plot_value(self,data):
        mapped_data = self.data_mapping(data)
        plt.figure(figsize=(5,5))
        plt.plot(mapped_data[:,0],mapped_data[:,1],"-o",color="red",linewidth=2)

        if(self.xaxis_type == AxisType.LOG):
            plt.xscale("log")
        if(self.yaxis_type == AxisType.LOG):
            plt.yscale("log")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Extract Data")
        plt.ylim(self.startY,self.endY)
        plt.xlim(self.startX,self.endX)
        plt.grid()
        plt.show()

d1 = DataExtractor()

def hex2rgb(hex_val):
    h = hex_val.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))





img = ImageGrab.grabclipboard()

d1.set_axis_type(xtype = AxisType.LINEAR,ytype = AxisType.LINEAR)
# img = Image.open('img/10.jpg')
color = input('input color:\n')
d1.set_fuzzy_color_range(hex2rgb(color))

xmin = float(input('minimal x:\n'))
xmax = float(input("maximum x:\n"))
ymin = float(input("minimal y:\n"))
ymax = float(input("maximum y:\n"))

d1.xaxis_type = AxisType.LOG if input("X Log?") == 'y' else AxisType.LINEAR
d1.yaxis_type = AxisType.LOG if input("Y Log?") == 'y' else AxisType.LINEAR

d1.set_axis_range([xmin,xmax,ymin,ymax])
# print(hex2rgb('#ae81ff'))
# d1.set_fuzzy_color_range([237, 34, 36])

d1.extract(img)
cv2.waitKey()