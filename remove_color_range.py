from PIL import Image
import cv2
import numpy

import matplotlib.pyplot as plt

# 18, 121, 190
_red_lim = [10,30]
_green_lim = [100,150]
_blue_lim = [180,220]

class DataExtractor:
    def __init__(self):
        self._red_lim = [0,20]
        self._green_lim = [0,20]
        self._blue_lim = [0,20]
        self._fuzzy_range = 100 #fuzzy range refers to picking range(0~255)
        self._step = 10
        self._x1 = 56
        self._x2 = 598
        self._y1 = 12
        self._y2 = 612
        self._filter_size = 2
        self.width = 0
        self.height = 0
        self.startX = -80#从datasheet读取的数据
        self.startY = 0 # 从datasheet读取的数据
        self.endX = 160
        self.endY = 2
        self.image_width = self._x2-self._x1
        self.image_height = self._y2-self._y1

    def set_fuzzy_color_range(self,pixdata):
        self._red_lim[0] = pixdata[0] - self._fuzzy_range if pixdata[0]>self._fuzzy_range else 0
        self._red_lim[1] = pixdata[0] + self._fuzzy_range if pixdata[0]<255-self._fuzzy_range else 255
        self._green_lim[0] = pixdata[1] - self._fuzzy_range if pixdata[1]>self._fuzzy_range else 0
        self._green_lim[1] = pixdata[1] + self._fuzzy_range if pixdata[1]<255-self._fuzzy_range else 255
        self._blue_lim[0] = pixdata[2] - self._fuzzy_range if pixdata[2]>self._fuzzy_range else 0
        self._blue_lim[1] = pixdata[2] + self._fuzzy_range if pixdata[2]<255-self._fuzzy_range else 255

    def check_valid(self,pixdata):
        if(pixdata[0]<self._red_lim[0] or pixdata[0]>self._red_lim[1]):
            return False
        if(pixdata[1]<self._green_lim[0] or pixdata[1]>self._green_lim[1]):
            return False
        if(pixdata[2]<self._blue_lim[0] or pixdata[2]>self._blue_lim[1]):
            return False
        return True

    def filter_color_range(self,image):
        img = image
        pixdata = img.load()
        original_image = numpy.array(img)#转换为cv2能用的格式
        cv2.imshow("original",original_image)
        self.width = img.size[0]
        self.height = img.size[1]

        for y in range(img.size[1]):
            for x in range(img.size[0]):
                # 二值化
                if not self.check_valid(pixdata[x,y]):
                    pixdata[x, y] = (255,255,255,255)
                else:
                    pixdata[x, y] = (0,0,0,0)
        image = numpy.array(img)#转换为cv2能用的格式
        image = image[:, :, :].copy()
        
        # result是一个y*x*color_channel的数组
        result = self.remove_grid(img)

        # 去除边界，但是目前这个写法可能不适合提取数据
        # result[:,0:self._x1] = 0
        # result[self._y2:self.height,:] = 0

        return result

    def remove_grid(self,pixdata):
        image = numpy.array(pixdata)
        image = image[:, :, :].copy()  
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#读取灰度
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove grid
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self._filter_size,self._filter_size))
        result = cv2.erode(thresh,horizontal_kernel)
        return result

    def extract(self,file_name):
        img = Image.open(file_name)
        img = img.convert("RGBA")
        binary_data = self.filter_color_range(img)
        # 去除边界
        binary_data_without_border = self.remove_boarder(binary_data)
        # cv2.imshow("border move",binary_data_without_border)

        extracted_data = self.extract_data(binary_data_without_border)

        self.plot_value(extracted_data[:,0],extracted_data[:,1])

    def remove_boarder(self,data):
        return data[self._y1:self._y2,self._x1:self._x2]
        
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

    def plot_value(self,x,y):
        plt.figure(figsize=(5,5))
        plt.plot(x/self.width*(self.endX-self.startX)+self.startX,y/self.height*(self.endY-self.startY)+self.startY,"-x",label="$sin(x)$",color="red",linewidth=2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Extract Data")
        plt.ylim(self.startY,self.endY)
        plt.xlim(self.startX,self.endX)
        plt.grid()
        plt.show()

        

d1 = DataExtractor()
d1.set_fuzzy_color_range([0, 0, 0])
d1.extract('img/4.png')
cv2.waitKey()
# 780*439