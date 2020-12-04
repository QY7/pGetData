from DataExtractor import DataExtractor,AxisType
from PIL import Image
from PIL import ImageGrab
from numpy import array,savetxt
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

    #定义槽函数
    def hello(self):
        self.lineEdit_3.setText("hello world")

    def extract_data(self):
        try:
            xmin = float(self.lineEdit.text())
            xmax = float(self.lineEdit_2.text())
            ymin = float(self.lineEdit_4.text())
            ymax = float(self.lineEdit_3.text())
        except:
            return
        d1 = DataExtractor()
        # #读取文件
        # img = Image.open('img/8.jpg')
        drawing_mode = False
        img = ImageGrab.grabclipboard()
        # 读取剪切板
        if len(array(img).shape)!=3:
            return

        d1.set_axis_type(xtype = AxisType.LINEAR,ytype = AxisType.LINEAR)

        # print("Step4. 输入坐标轴类型\n")
        d1.xaxis_type = AxisType.LOG if self.checkBox_2.isChecked() else AxisType.LINEAR
        d1.yaxis_type = AxisType.LOG if self.checkBox.isChecked() else AxisType.LINEAR


        d1.set_axis_range([xmin,xmax,ymin,ymax])
        # print("Step5. 单击选择颜色\n")
        result = d1.extract(img)
        savetxt("..\\test.txt",result)
        plot_value(result,xaxis_type=d1.xaxis_type,yaxis_type=d1.yaxis_type)
        cv2.waitKey()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #MainWindow = QMainWindow()
    window = mywindow()
    window.show()
    window.setWindowTitle("DataExtractor")
    window.setWindowIcon(QIcon('img/computer.ico'))
    window.pushButton.clicked.connect(window.extract_data)
    sys.exit(app.exec_())