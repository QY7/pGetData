from DataExtractor import DataExtractor,AxisType
from PIL import Image
from PIL import ImageGrab
from numpy import array,savetxt
import cv2
import matplotlib.pyplot as plt
from ui import Ui_MainWindow

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

if __name__ == "__main__":
    # print("Step1. 截图，完成输入回车")
    # input()
    # d1 = DataExtractor()
    # # #读取文件
    # # img = Image.open('img/8.jpg')

    # drawing_mode = False
    # print("Step2. 读取剪切板中")
    # img = ImageGrab.grabclipboard()
    # # 读取剪切板
    # if len(array(img).shape)!=3:
    #     print("剪切板无图片,程序退出")
    #     exit(0)
    # # if(img.shape == (3,3,))

    # d1.set_axis_type(xtype = AxisType.LINEAR,ytype = AxisType.LINEAR)

    # # color = input('input color:\n')
    # # color = color if(color != '') else '#000000'

    # # d1.set_fuzzy_color_range(hex2rgb(color))
    # print("Step3. 输入坐标轴范围\n")
    # xmin = float(input('minimal x:\n'))
    # xmax = float(input("maximum x:\n"))
    # ymin = float(input("minimal y:\n"))
    # ymax = float(input("maximum y:\n"))

    # print("Step4. 输入坐标轴类型\n")
    # d1.xaxis_type = AxisType.LOG if input("X Log? y/n") == 'y' else AxisType.LINEAR
    # d1.yaxis_type = AxisType.LOG if input("Y Log? y/n" ) == 'y' else AxisType.LINEAR

    # d1.set_axis_range([xmin,xmax,ymin,ymax])

    # print("Step5. 单击选择颜色\n")
    # result = d1.extract(img)
    # savetxt("..\\test.txt",result)
    # plot_value(result,xaxis_type=d1.xaxis_type,yaxis_type=d1.yaxis_type)

    # cv2.waitKey()
    app = QApplication(sys.argv) 
	form = QMainWindow()
	w = Ui_MainWindow()
	w.setupUi(form)
	form.show()
	sys.exit(app.exec_())