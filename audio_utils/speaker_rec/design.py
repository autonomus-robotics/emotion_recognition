# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 348)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 29, 581, 261))
        self.textEdit.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.textEdit.setObjectName("textEdit")
        self.FIO_Edit = QtWidgets.QTextEdit(self.centralwidget)
        self.FIO_Edit.setEnabled(True)
        self.FIO_Edit.setGeometry(QtCore.QRect(180, 300, 411, 31))
        self.FIO_Edit.setObjectName("FIO_Edit")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(9, 300, 158, 31))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Recognize_Button = QtWidgets.QPushButton(self.widget)
        self.Recognize_Button.setObjectName("Recognize_Button")
        self.horizontalLayout.addWidget(self.Recognize_Button)
        self.Add_Button = QtWidgets.QPushButton(self.widget)
        self.Add_Button.setAcceptDrops(False)
        self.Add_Button.setObjectName("Add_Button")
        self.horizontalLayout.addWidget(self.Add_Button)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Recognize_Button.setText(_translate("MainWindow", "Recognize"))
        self.Add_Button.setText(_translate("MainWindow", "Add"))

