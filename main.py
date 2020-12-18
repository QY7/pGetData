# from DataExtractor import DataExtractor,AxisType
from PIL import Image
from PIL import ImageGrab
from numpy import array,savetxt,tile,newaxis
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ui import Ui_MainWindow
import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from enum import Enum


class AxisType(Enum):
    LINEAR = 1
    LOG = 2


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
            
# def hex2rgb(hex_val):
#     h = hex_val.lstrip('#')
#     return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

class mywindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.reset()
        
    def load_img(self):
        # self.__init__()
        try:
            img_clip = ImageGrab.grabclipboard()
            img = cv2.cvtColor(np.array(img_clip),cv2.COLOR_RGB2BGR)
        except TypeError as e:
            print(e)
            return
        # try:
        # except:
        #     return
        if len(array(img).shape)!=3:
            return

        if(np.array_equal(img[:,:,0],img[:,:,1])):
            self.color_mode = False
        else:
            self.color_mode = True
        self.load_state = True
        self.original_img = img
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        # im_np = np.transpose(img,(1,0,2)).copy()
        self.update_img(self.original_img)

        self.label_info.setText("STEP2: Fill in axis")
        self.operation_stage = 1



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

    def update_img(self,img):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_BGR888)
        self.label_img.setPixmap(QtGui.QPixmap(qImg))

    def read_config(self):
        try:
            self.startX = float(self.lineEdit_xmin.text())
            self.endX = float(self.lineEdit_xmax.text())
            self.startY = float(self.lineEdit_ymin.text())
            self.endY = float(self.lineEdit_ymax.text())
            self.xaxis_type = AxisType.LOG if self.radio_logx.isChecked() else AxisType.LINEAR
            self.yaxis_type = AxisType.LOG if self.radio_logy.isChecked() else AxisType.LINEAR

            assert (self.endX>self.startX)
            assert (self.endY >self.startY)
            assert ((self.xaxis_type is AxisType.LINEAR) or (self.xaxis_type is AxisType.LOG and (self.startX>0)))
            assert ((self.yaxis_type is AxisType.LINEAR) or (self.yaxis_type is AxisType.LOG and (self.startY>0)))

            return True
        except:
            QMessageBox.warning(self,"Warning","Invalid input!")
            return False

        
    def set_fuzzy_color_range(self,pixdata):
        color = cv2.cvtColor(np.uint8([[pixdata]]),cv2.COLOR_BGR2HSV)
        # print(color)
        self.lower_color_lim = array([color[0][0][0],20,20])
        self.higher_color_lim = array([color[0][0][0],255,255])
    def get_color_mask(self,img):
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,self.lower_color_lim,self.higher_color_lim)
        return mask

    def erase_on(self,event):
        self.drawing_mode = True
    def erase_off(self,event):
        self.drawing_mode = False
    def erasing(self,event):
        x = event.pos().x()
        y = event.pos().y()

        if(x<0):
            x = 0
        elif (x>=self.image_width-1):
            x = self.image_width-1
        if(y<0):
            y = 0
        elif(y>self.image_height-1):
            y = self.image_height-1

        if self.drawing_mode:
            y_range_up = y-self.erase_range
            y_range_down = y+self.erase_range
            x_range_left = x-self.erase_range
            x_range_right = x+self.erase_range

            y_range_up = y_range_up if(y_range_up>0) else 0
            y_range_down = y_range_down if(y_range_down<self.image_height) else self.image_height-1
            x_range_left = x_range_left if(x_range_left>0) else 0
            x_range_right = x_range_right if(x_range_right<=self.image_width) else self.image_width-1

            self.mask[y_range_up:y_range_down,x_range_left:x_range_right]=0
            masked_img = tile(self.mask[:,:,newaxis],3)
            blender=cv2.addWeighted(self.original_img,self._blend_transparency,masked_img,1,0)
            self.update_img(blender)


    def extract_data(self,data):
        result = []
        # data = data.T

        for i in range(0,self.image_width,self._step):
            # 如果有黑色像素，取中值
            # 注意行列和矩阵的维度的对应
            # 矩阵中，第一维为行，第二为列。所以对x坐标进行循环， 就应该取出对应的列的所有行
            if(255 in data[:,i]):
                ldx = self.image_height-np.median(np.argwhere(data[:,i]==255))
                result.append([i,ldx])

        # print(array(result))
        return array(result)

    def pick_color(self,event):
        x = event.pos().x()
        y = event.pos().y()
        self.color_set = self.original_img[y][x]
        self.pushButton_start.setEnabled(True)
        print("当前选择颜色为:")
        print(self.color_set)


    def reset(self):
        self.label_img.setText("No Image")
        self.load_state = False
        self.result = None

        self.lower_color_lim = [0,0,0]
        self.higher_color_lim = [255,255,255]

        self._fuzzy_range = 50 #fuzzy range refers to picking range(0~255)
        self._step = 10
        self._blend_transparency = 0.3
        self.grid_size = 3
        self.image_width = 0
        self.image_height = 0
        self.startX = 0 #x坐标最小
        self.startY = 1 #x坐标最大
        self.endX =0 #y坐标最小
        self.endY = 1 #y坐标最大

        self.xaxis_type = AxisType.LINEAR
        self.yaxis_type = AxisType.LINEAR

        self.original_img  = 0
        self.filtered_img = 0
        self.mask = 0

        self.color_mode = True

        self.color_set = None
        self.erase_range = 5
        self.drawing_mode = False

        self.operation_stage = 0
        self.label_info.setText("STEP1: Load from clipboard")
    def data_mapping(self,data):
        if(self.xaxis_type == AxisType.LINEAR):
            data[:,0] = data[:,0]/self.image_width*(self.endX-self.startX)+self.startX
        else:
            data[:,0] = np.power(10,(data[:,0]/self.image_width*(np.log10(self.endX)-np.log10(self.startX))+np.log10(self.startX)))
        if(self.yaxis_type == AxisType.LINEAR):
            data[:,1] = data[:,1]/self.image_height*(self.endY-self.startY)+self.startY
        else:
            data[:,1] = np.power(10,(data[:,1]/self.image_height*(np.log10(self.endY)-np.log10(self.startY))+np.log10(self.startY)))
        return data
        
    def color_extractor(self):
        # if(self.checkBox_black.isChecked()):
        # self.extract_data(mode=False)
        if(self.operation_stage == 1):
            if(not self.read_config()):
                return
            else:
                if(self.color_mode):
                    self.operation_stage = 2
                    # 彩色模式，stage1进行挑选颜色
                    self.label_img.mousePressEvent = self.pick_color
                    self.pushButton_start.setEnabled(False)
                    self.label_info.setText("STEP3: Pick color and hit next")
                else:
                    # 黑色模式
                    self.operation_stage = 2
                    self.label_info.setText("STEP3: Hit next")
            
            return
        if(self.operation_stage ==2):
            # stage1筛选颜色阶段
            if(self.color_mode):
                # 彩色模式，stage1进行挑选颜色
                self.set_fuzzy_color_range(self.color_set)
                self.mask = self.get_color_mask(self.original_img)
            else:
                # 黑色模式
                self.mask = self.rm_grid(int(0.4*self.image_width),int(0.4*self.image_height))
        
            self.filtered_img = tile(self.mask[:,:,newaxis],3)
            blender=cv2.addWeighted(self.original_img,self._blend_transparency,self.filtered_img,1,0)
            self.update_img(blender)
        
            self.label_info.setText("STEP4: erase noise")
            self.label_img.mousePressEvent = self.erase_on
            self.label_img.mouseReleaseEvent = self.erase_off
            self.label_img.mouseMoveEvent = self.erasing
            
            self.operation_stage = 3

        elif(self.operation_stage == 3):
            self.label_img.mousePressEvent = None
            self.label_img.mouseReleaseEvent = None
            self.label_img.mouseMoveEvent = None
            extracted_data = self.extract_data(self.mask)
            # 数据点映射坐标
            mapped_data = self.data_mapping(extracted_data)
            plot_value(mapped_data)


    def export_data(self):
        try:
            filename=QFileDialog.getSaveFileName(self,'save file',filter="Txt files(*.txt)")
            savetxt(filename[0],self.result,delimiter=';')
        except:
            return

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #MainWindow = QMainWindow()
    window = mywindow()
    window.show()
    window.setWindowTitle("DataExtractor")
    window.setWindowIcon(QIcon('img/computer.ico'))

    window.pushButton_start.clicked.connect(window.color_extractor)
    window.pushButton_reset.clicked.connect(window.reset)
    window.pushButton_load.clicked.connect(window.load_img)
    # window.Eraser_size.valueChanged.connect(window.eraser_size_change)
    # window.Grid_size.valueChanged.connect(window.grid_size_change)
    sys.exit(app.exec_())