# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\main.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(905, 692)
        MainWindow.setStyleSheet("*{\n"
"}\n"
"QCheckBox{\n"
"font-family:Arial;\n"
"font-size:15px;\n"
"font-weight:bold;\n"
"}\n"
"QRadioButton{\n"
"font-family:Arial;\n"
"font-size:20px;\n"
"font-weight:bold;\n"
"}\n"
"QLabel{\n"
"font-family:Arial;\n"
"color:#222\n"
"}\n"
"QPushButton\n"
"{\n"
"color:white;\n"
"background:    #aaa;\n"
"font-weight:bold;\n"
"font-size:15px;\n"
"border-radius:5px;\n"
"font-family:Arial;\n"
"}\n"
"QPushButton:hover{\n"
"background:    #3498db;\n"
"}\n"
"QLineEdit{\n"
"border-radius:4px;\n"
"}\n"
"#label_info{\n"
"color:white;\n"
"background:#ef5350;\n"
"padding:10px\n"
"}\n"
"#label_img{\n"
"cursor:crosshair;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 2, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 0, 1, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 2, 1, 1, 1)
        self.label_img = QtWidgets.QLabel(self.centralwidget)
        self.label_img.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.label_img.setObjectName("label_img")
        self.gridLayout.addWidget(self.label_img, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(0, 0))
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_12.addWidget(self.label_3)
        self.lineEdit_xmin = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_xmin.sizePolicy().hasHeightForWidth())
        self.lineEdit_xmin.setSizePolicy(sizePolicy)
        self.lineEdit_xmin.setMinimumSize(QtCore.QSize(0, 20))
        self.lineEdit_xmin.setMaximumSize(QtCore.QSize(16777215, 30))
        self.lineEdit_xmin.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.lineEdit_xmin.setFont(font)
        self.lineEdit_xmin.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_xmin.setObjectName("lineEdit_xmin")
        self.horizontalLayout_12.addWidget(self.lineEdit_xmin)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_12.addWidget(self.label_2)
        self.lineEdit_xmax = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_xmax.sizePolicy().hasHeightForWidth())
        self.lineEdit_xmax.setSizePolicy(sizePolicy)
        self.lineEdit_xmax.setMinimumSize(QtCore.QSize(0, 20))
        self.lineEdit_xmax.setMaximumSize(QtCore.QSize(16777215, 30))
        self.lineEdit_xmax.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.lineEdit_xmax.setFont(font)
        self.lineEdit_xmax.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_xmax.setObjectName("lineEdit_xmax")
        self.horizontalLayout_12.addWidget(self.lineEdit_xmax)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem4)
        self.checkBox_logx = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox_logx.sizePolicy().hasHeightForWidth())
        self.checkBox_logx.setSizePolicy(sizePolicy)
        self.checkBox_logx.setObjectName("checkBox_logx")
        self.horizontalLayout_12.addWidget(self.checkBox_logx)
        self.verticalLayout.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setMinimumSize(QtCore.QSize(0, 0))
        self.label_6.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_5.addWidget(self.label_6)
        self.lineEdit_ymin = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_ymin.sizePolicy().hasHeightForWidth())
        self.lineEdit_ymin.setSizePolicy(sizePolicy)
        self.lineEdit_ymin.setMinimumSize(QtCore.QSize(0, 20))
        self.lineEdit_ymin.setMaximumSize(QtCore.QSize(16777215, 30))
        self.lineEdit_ymin.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.lineEdit_ymin.setFont(font)
        self.lineEdit_ymin.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_ymin.setObjectName("lineEdit_ymin")
        self.horizontalLayout_5.addWidget(self.lineEdit_ymin)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_5.addWidget(self.label_4)
        self.lineEdit_ymax = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_ymax.sizePolicy().hasHeightForWidth())
        self.lineEdit_ymax.setSizePolicy(sizePolicy)
        self.lineEdit_ymax.setMinimumSize(QtCore.QSize(0, 20))
        self.lineEdit_ymax.setMaximumSize(QtCore.QSize(16777215, 30))
        self.lineEdit_ymax.setSizeIncrement(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Arial")
        self.lineEdit_ymax.setFont(font)
        self.lineEdit_ymax.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_ymax.setObjectName("lineEdit_ymax")
        self.horizontalLayout_5.addWidget(self.lineEdit_ymax)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem5)
        self.checkBox_logy = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.checkBox_logy.sizePolicy().hasHeightForWidth())
        self.checkBox_logy.setSizePolicy(sizePolicy)
        self.checkBox_logy.setObjectName("checkBox_logy")
        self.horizontalLayout_5.addWidget(self.checkBox_logy)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 1, 0, 1, 1)
        self.horizontalSlider_eraser = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_eraser.setMinimum(1)
        self.horizontalSlider_eraser.setMaximum(100)
        self.horizontalSlider_eraser.setProperty("value", 20)
        self.horizontalSlider_eraser.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_eraser.setObjectName("horizontalSlider_eraser")
        self.gridLayout_2.addWidget(self.horizontalSlider_eraser, 0, 1, 1, 1)
        self.horizontalSlider_morph = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_morph.setMinimum(1)
        self.horizontalSlider_morph.setMaximum(255)
        self.horizontalSlider_morph.setProperty("value", 100)
        self.horizontalSlider_morph.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_morph.setObjectName("horizontalSlider_morph")
        self.gridLayout_2.addWidget(self.horizontalSlider_morph, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem6)
        self.radioButton_black = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_black.setObjectName("radioButton_black")
        self.horizontalLayout.addWidget(self.radioButton_black)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem7)
        self.radioButton_color = QtWidgets.QRadioButton(self.centralwidget)
        self.radioButton_color.setObjectName("radioButton_color")
        self.horizontalLayout.addWidget(self.radioButton_color)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem8)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.label_info = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_info.setFont(font)
        self.label_info.setAlignment(QtCore.Qt.AlignCenter)
        self.label_info.setObjectName("label_info")
        self.verticalLayout.addWidget(self.label_info)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.pushButton_load = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.pushButton_load.sizePolicy().hasHeightForWidth())
        self.pushButton_load.setSizePolicy(sizePolicy)
        self.pushButton_load.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_load.setFont(font)
        self.pushButton_load.setObjectName("pushButton_load")
        self.horizontalLayout_10.addWidget(self.pushButton_load)
        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.pushButton_start.sizePolicy().hasHeightForWidth())
        self.pushButton_start.setSizePolicy(sizePolicy)
        self.pushButton_start.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setStyleSheet("")
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout_10.addWidget(self.pushButton_start)
        self.pushButton_reset = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(5)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.pushButton_reset.sizePolicy().hasHeightForWidth())
        self.pushButton_reset.setSizePolicy(sizePolicy)
        self.pushButton_reset.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(-1)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_reset.setFont(font)
        self.pushButton_reset.setStyleSheet("")
        self.pushButton_reset.setObjectName("pushButton_reset")
        self.horizontalLayout_10.addWidget(self.pushButton_reset)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 905, 22))
        self.menubar.setObjectName("menubar")
        self.menuhelp = QtWidgets.QMenu(self.menubar)
        self.menuhelp.setObjectName("menuhelp")
        self.menufile = QtWidgets.QMenu(self.menubar)
        self.menufile.setObjectName("menufile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExport = QtWidgets.QAction(MainWindow)
        self.actionExport.setObjectName("actionExport")
        self.actionImport = QtWidgets.QAction(MainWindow)
        self.actionImport.setObjectName("actionImport")
        self.menuhelp.addAction(self.actionAbout)
        self.menufile.addAction(self.actionExport)
        self.menufile.addAction(self.actionImport)
        self.menubar.addAction(self.menufile.menuAction())
        self.menubar.addAction(self.menuhelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_img.setText(_translate("MainWindow", "Label_img"))
        self.label_3.setText(_translate("MainWindow", "xmin"))
        self.label_2.setText(_translate("MainWindow", "xmax"))
        self.checkBox_logx.setText(_translate("MainWindow", "Logx"))
        self.label_6.setText(_translate("MainWindow", "ymin"))
        self.label_4.setText(_translate("MainWindow", "ymax"))
        self.checkBox_logy.setText(_translate("MainWindow", "Logy"))
        self.label.setText(_translate("MainWindow", "Eraser"))
        self.label_5.setText(_translate("MainWindow", "Morph"))
        self.radioButton_black.setText(_translate("MainWindow", "Black"))
        self.radioButton_color.setText(_translate("MainWindow", "Color"))
        self.label_info.setText(_translate("MainWindow", "STEP1: Clip graph and fill in data"))
        self.pushButton_load.setText(_translate("MainWindow", "Load"))
        self.pushButton_start.setText(_translate("MainWindow", "Next"))
        self.pushButton_reset.setText(_translate("MainWindow", "Reset"))
        self.menuhelp.setTitle(_translate("MainWindow", "help"))
        self.menufile.setTitle(_translate("MainWindow", "file"))
        self.actionAbout.setText(_translate("MainWindow", "V1.0"))
        self.actionExport.setText(_translate("MainWindow", "Export"))
        self.actionImport.setText(_translate("MainWindow", "Import"))
# import main_rc
