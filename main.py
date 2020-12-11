from DataExtractor import DataExtractor,AxisType
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
        self.extractor = DataExtractor()
        self.color_mode = False
        self.load_state = False
        self.result = None

    def load_img(self):
        try:
            img = np.array(ImageGrab.grabclipboard())
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        except TypeError as e:
            print(e)
        # try:
        # except:
        #     return
        if len(array(img).shape)!=3:
            return False
        self.load_state = True
        self.extractor.original_img = img
        self.extractor.image_width = img.shape[1]
        self.extractor.image_height = img.shape[0]
        return True

    def extract_data(self,mode):
        try:
            xmin = float(self.lineEdit.text())
            xmax = float(self.lineEdit_2.text())
            ymin = float(self.lineEdit_4.text())
            ymax = float(self.lineEdit_3.text())
            self.extractor.xaxis_type = AxisType.LOG if self.checkBox_2.isChecked() else AxisType.LINEAR
            self.extractor.yaxis_type = AxisType.LOG if self.checkBox.isChecked() else AxisType.LINEAR
        except:
            return
        # xmin = 1
        # xmax = 2
        # ymin = 3
        # ymax = 4
        drawing_mode = False
        if(self.load_img()):
            self.extractor.set_axis_range([xmin,xmax,ymin,ymax])
            self.result =self.extractor.extract(mode)
            
            plot_value(self.result,xaxis_type=self.extractor.xaxis_type,yaxis_type=self.extractor.yaxis_type)
            cv2.waitKey()

    def color_extractor(self):
        if(self.checkBox_black.isChecked()):
            self.extract_data(mode=False)
        else:
            self.extract_data(mode=True)

    def eraser_size_change(self):
        self.extractor.erase_range = self.Eraser_size.value()

    def export_data(self):
        try:
            filename=QFileDialog.getSaveFileName(self,'save file',filter="Txt files(*.txt)")
            savetxt(filename,self.result,delimiter='; ')
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
    window.pushButton_export.clicked.connect(window.export_data)

    window.Eraser_size.valueChanged.connect(window.eraser_size_change)
    # window.Grid_size.valueChanged.connect(window.grid_size_change)
    sys.exit(app.exec_())