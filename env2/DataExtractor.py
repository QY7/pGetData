from PIL import Image
from PIL import ImageGrab
import cv2
# from cv2 import cvtColor,imshow,threshold,erode,getStructuringElement,MORPH_RECT,THRESH_OTSU,THRESH_BINARY_INV,COLOR_BGR2GRAY
from numpy import array,median,argwhere,power,tile,newaxis,savetxt
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
        self._blend_transparency = 0.2
        # self._x1 = 0
        # self._x2 = 580
        # self._y1 = 13
        # self._y2 = 430
        self._filter_size = 1
        self.width = 0
        self.height = 0
        self.startX = 0.5#从datasheet读取的数据
        self.startY = 1 # 从datasheet读取的数据
        self.endX =100
        self.endY = 4000

        self.xaxis_type = AxisType.LINEAR
        self.yaxis_type = AxisType.LOG

        self.original_img  = 0
        self.color_set = None
        self.erase_range = 15
        

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
        original_image = array(img)#转换为cv2能用的格式
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
                    
        image = array(img)#转换为cv2能用的格式
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
        image = array(pixdata)
        image = image[:, :, :].copy()  
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#读取灰度
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove grid
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._filter_size,self._filter_size))
        result = cv2.erode(thresh,horizontal_kernel)
        return result

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
        global drawing_mode
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_mode = True

        if event == cv2.EVENT_MOUSEMOVE and drawing_mode:
            self.binary_data_without_border[y-self.erase_range:y+self.erase_range,x-self.erase_range:x+self.erase_range]=0
            blender=cv2.addWeighted(self.original_img,self._blend_transparency,tile(self.binary_data_without_border[:,:,newaxis],3),0.9,0)
            cv2.imshow("masked image",blender)
        

        if event == cv2.EVENT_LBUTTONUP:
            drawing_mode = False
            # cv2.imshow("extracted image",self.binary_data_without_border)
        # mouseX,mouseY = x,y
    def pick_color(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_mode = True
            self.color_set = self.original_img[y][x]
            print("The color selected is:\n")
            print(self.color_set)

    def extract(self,img):
        # self.width = img.size[0]
        # self.height = img.size[1]
        # 根据颜色范围选择要用到的数据
        self.original_img = array(img)#转换为cv2能用的格式
        self.original_img  = self.original_img [:, :, :].copy()

        self.original_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)

        # 显示原图，让用户选择颜色
        cv2.namedWindow("original image")
        cv2.moveWindow("original image", 40,30)  # Move it to (40,30)
        cv2.imshow("original image",self.original_img)
        cv2.setMouseCallback("original image",self.pick_color)

        # 检测键盘输入，如果键盘输入回车，进入下一步
        while(1):
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == 13:
                # 直到检测到颜色输入为止
                if(self.color_set is not None):
                    d1.set_fuzzy_color_range(self.color_set[::-1])
                    cv2.destroyWindow("original image")
                else:
                    continue
                break
        # 筛选颜色
        binary_data = self.filter_color_range(img)
        # 去除边界
        self.binary_data_without_border = binary_data
        # binary_data_without_border = self.remove_boarder(binary_data,[self._x1,self._x2,self._y1,self._y2])
        # 提取数据点

        # 混合原图和筛选的图
        blender=cv2.addWeighted(self.original_img,self._blend_transparency,tile(self.binary_data_without_border[:,:,newaxis],3),0.9,0)
        
        cv2.namedWindow("masked image")
        cv2.moveWindow("masked image", 40,30)  # Move it to (40,30)
        cv2.imshow("masked image",blender)

        # 让用户去除不要的数据点
        cv2.setMouseCallback('masked image',self.erase_noise)
        while(1):
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break

            elif k == 13:
                # cv2.imshow("after",self.binary_data_without_border)
                cv2.destroyWindow("masked image")
                extracted_data = self.extract_data(self.binary_data_without_border)
                # 数据点映射坐标
                mapped_data = self.data_mapping(extracted_data)
                return mapped_data

def plot_value(data,xlim = None, ylim = None,xaxis_type = AxisType.LINEAR,yaxis_type = AxisType.LINEAR):
    plt.figure(figsize=(5,5))
    plt.plot(data[:,0],data[:,1],"-o",color="red",linewidth=2)

    if(xaxis_type == AxisType.LOG):
        plt.xscale("log")
    if(yaxis_type == AxisType.LOG):
        plt.yscale("log")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Extract Data")
    if(xlim):
        plt.ylim(self.startY,self.endY)
    if(ylim):
        plt.xlim(self.startX,self.endX)
    plt.grid()
    plt.show()
            
def hex2rgb(hex_val):
    h = hex_val.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

print("Step1. take a screenshot")
input()
d1 = DataExtractor()
# #读取文件
# img = Image.open('img/8.jpg')

drawing_mode = False
print("Step2. reading from the clipboard")
img = ImageGrab.grabclipboard()
# 读取剪切板
if len(array(img).shape)!=3:
    print("Nothing on the clipboard, exiting")
    exit(0)
# if(img.shape == (3,3,))

d1.set_axis_type(xtype = AxisType.LINEAR,ytype = AxisType.LINEAR)

# color = input('input color:\n')
# color = color if(color != '') else '#000000'

# d1.set_fuzzy_color_range(hex2rgb(color))
print("Step3. Input the scale of the axis\n")
xmin = float(input('minimal x:\n'))
xmax = float(input("maximum x:\n"))
ymin = float(input("minimal y:\n"))
ymax = float(input("maximum y:\n"))

print("Step4. Input the type of the axis\n")
d1.xaxis_type = AxisType.LOG if input("X Log? y/n") == 'y' else AxisType.LINEAR
d1.yaxis_type = AxisType.LOG if input("Y Log? y/n" ) == 'y' else AxisType.LINEAR

d1.set_axis_range([xmin,xmax,ymin,ymax])

print("Step5. Choose curve by its color\n")
result = d1.extract(img)
savetxt("..\\test.txt",result)
plot_value(result,xaxis_type=d1.xaxis_type,yaxis_type=d1.yaxis_type)

cv2.waitKey()