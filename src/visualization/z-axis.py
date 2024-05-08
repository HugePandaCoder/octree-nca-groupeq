import sys
from typing import Tuple
from PyQt6 import QtGui

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

import numpy as np
from PIL import Image

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.title = "Z-Axis: out_0"
        self.setWindowTitle(self.title)

        self.picture = 0

        self.layout = QGridLayout()
        self.label = QLabel(self)
        self.pixmap = QPixmap('Bilderchen/out_0.png')
        self.label.setPixmap(self.pixmap)
        self.label.setScaledContents(True)

        # self.label_t = QLabel(self)
        # self.label_t.setText('out_0')
        # self.label_t.resize(20, 20)

        # self.label_t.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # self.layout.addWidget(self.label_t)
        self.layout.addWidget(self.label)

        self.button_r = QPushButton(self)
        self.button_r.setText("z+")
        def button_r_clicked():
            self.picture = self.picture + 1
            if self.picture == 52:
                self.picture = 0
            pic = 'Bilderchen/out_' + str(self.picture) + '.png'
            self.label.setPixmap(QPixmap(pic))
            self.title = "Z-Axis: out_" + str(self.picture)
            self.setWindowTitle(self.title)


        self.button_l = QPushButton(self)
        self.button_l.setText("z-")
        def button_l_clicked():
            self.picture = self.picture - 1
            if self.picture < 0:
                self.picture = 51
            pic = 'Bilderchen/out_' + str(self.picture) + '.png'
            self.label.setPixmap(QPixmap(pic))
            self.title = "Z-Axis: out_" + str(self.picture)
            self.setWindowTitle(self.title)

        self.button_r.clicked.connect(button_r_clicked)
        self.button_l.clicked.connect(button_l_clicked)

        self.layout.addWidget(self.button_r)
        self.layout.addWidget(self.button_l)

        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        self.resize(640, 640)  # self.resize(pixmap.width(), pixmap.height()) 

if  __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
