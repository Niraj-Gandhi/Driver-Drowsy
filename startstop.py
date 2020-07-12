# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'startstop.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import drowsy
class Ui_Startstop(object):
    def setupUi(self, Startstop):
        Startstop.setObjectName("Startstop")
        Startstop.resize(244, 335)
        self.centralwidget = QtWidgets.QWidget(Startstop)
        self.centralwidget.setObjectName("centralwidget")
        self.start = QtWidgets.QPushButton(self.centralwidget)
        self.start.setGeometry(QtCore.QRect(30, 100, 181, 51))
        self.start.setObjectName("start")
        self.stop = QtWidgets.QPushButton(self.centralwidget)
        self.stop.setGeometry(QtCore.QRect(30, 190, 181, 51))
        self.stop.setObjectName("stop")
        Startstop.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Startstop)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 244, 21))
        self.menubar.setObjectName("menubar")
        Startstop.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Startstop)
        self.statusbar.setObjectName("statusbar")
        Startstop.setStatusBar(self.statusbar)

        self.retranslateUi(Startstop)
        QtCore.QMetaObject.connectSlotsByName(Startstop)
        self.start.clicked.connect(lambda :drowsy.drowsydetect())
        self.stop.clicked.connect(lambda :self.exitfunc(Startstop))

    def retranslateUi(self, Startstop):
        _translate = QtCore.QCoreApplication.translate
        Startstop.setWindowTitle(_translate("Startstop", "MainWindow"))
        self.start.setText(_translate("Startstop", "START"))
        self.stop.setText(_translate("Startstop", "STOP"))
    def exitfunc(self,Startstop):
        drowsy.change()
        Startstop.close()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Startstop = QtWidgets.QMainWindow()
    ui = Ui_Startstop()
    ui.setupUi(Startstop)
    Startstop.show()
    sys.exit(app.exec_())
