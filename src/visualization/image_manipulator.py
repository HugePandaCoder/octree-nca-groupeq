import sys
import os
from typing import Tuple

from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

import numpy as np
from PIL import Image

from math import ceil

class ImageCanvas(QLabel):

    def __init__(self, img_path: str = None, init_img: QPixmap = None):
        """
            Backbone of image editor. Keeps track of mouse movements 
            and clicks to manipulate the canvas.
        """
        super().__init__()

        self.scale = 12

        if img_path is not None:
            self.background = QImage(img_path)
        elif init_img is not None:
            self.background = init_img.toImage()
        else:
            raise RuntimeError("ImageCanvas needs either image path or pixmap")

        self.setFixedSize(self.background.size() * self.scale)

        self.overlay = QPixmap(self.size())
        self.overlay.fill(QColorConstants.Transparent)

        self.setMouseTracking(True)

        self.dragging = (False, QPoint(- 1, -1))

        self.modified = True
        self.pen_mode = False
        self.color = 0
        self.update()

    def update_color(self, c):
        self.color = c

    def update_scale(self, sc):
        """
            Upates canvas scale.
        """
        self.scale = sc
        self.setFixedSize(self.background.size() * self.scale)
        self.overlay = QPixmap(self.size())
        self.modified = True

    def set_pen_mode(self, mode):
        self.pen_mode = mode

    def update_background(self, image_path: str):
        """
            Updates canvas background with image from path.
        """
        self.background = QImage(image_path)
        self.setFixedSize(self.background.size() * self.scale)
        self.update()

    def paintEvent(self, a0: QPaintEvent) -> None:
        super().paintEvent(a0)
        
        if self.modified:
            self.__update_pixmap()
            self.modified = False

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        """
            Tracks mouse movements over the canvas and draws a box
            around the currently hovered pixel.
        """
        super().mouseMoveEvent(ev)
        if not self.modified:
            self.overlay.fill(QColorConstants.Transparent)

        x = ev.pos().x()
        y = ev.pos().y()

        pix_x = int(x / self.scale)
        pix_y = int(y / self.scale)

        #pen for drawing while dragging
        if self.pen_mode:
            if ev.buttons() & Qt.MouseButton.LeftButton:
                self.background.setPixelColor(pix_x, pix_y, QColor(self.color, self.color, self.color))
                
        self.__draw_pixel_selector(pix_x, pix_y)
        self.modified = True
        self.update()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        """
            Edits currently hovered pixel with pen color. Also registers 
            drag movement starting points.
        """
        super().mousePressEvent(ev)
        x = ev.pos().x()
        y = ev.pos().y()
        pix_x = int(x / self.scale)
        pix_y = int(y / self.scale)
        # print(f"Set {pix_x},{pix_y}")
        self.background.setPixelColor(pix_x, pix_y, QColor(self.color, self.color, self.color))

        if ev.buttons() & Qt.MouseButton.RightButton:
            print(f"Start drag at x:{ev.pos().x()},y:{ev.pos().y()}")
            self.dragging = (True, ev.pos())

        self.modified = True
        self.update()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        """
            Used to edit multiple pixels which were covered by a mouse drag.
        """
        super().mouseReleaseEvent(ev)

        rel_pos = ev.pos()

        if self.dragging[0]:
            if not self.modified:
                self.overlay.fill(QColorConstants.Transparent)

            print(f"Stop drag at x:{rel_pos.x()},y:{rel_pos.y()}")
            drag_pos = self.dragging[1]
            topleft_x = min(drag_pos.x(), rel_pos.x())
            topleft_y = min(drag_pos.y(), rel_pos.y())
            width = abs(drag_pos.x() - rel_pos.x())
            height = abs(drag_pos.y() - rel_pos.y())

            drag_rect = QRect(topleft_x, topleft_y, width, height)
            print(f"Drag rect:{drag_rect}")
            #draw drag rect
            pix_x = int(topleft_x / self.scale)
            pix_y = int(topleft_y / self.scale)
            pix_width = ceil(((topleft_x - pix_x*self.scale) + width) / self.scale)
            pix_height = ceil(((topleft_y - pix_y*self.scale) + height) / self.scale)

            if pix_height > 1 and pix_width > 1:
                line_width = 4

                for i in range(pix_x, pix_x+pix_width):
                    for j in range(pix_y, pix_y+pix_height):
                        self.background.setPixelColor(i, j, QColor(self.color, self.color, self.color))

                print(f"pix width:{pix_width}, pix height:{pix_height}")
                print(f"Reconstructed rect: {QRect(pix_x*self.scale - int(line_width/2), pix_y*self.scale- int(line_width/2), self.scale*pix_width+line_width, self.scale*pix_height+line_width)}")

                overlay_painter = QPainter()
                overlay_painter.begin(self.overlay)
                pen = overlay_painter.pen()
                pen.setWidth(line_width)
                pen.setColor(QColorConstants.Red)
                overlay_painter.setPen(pen)
                overlay_painter.drawRect(pix_x*self.scale - int(line_width/2), pix_y*self.scale- int(line_width/2), self.scale*pix_width+line_width, self.scale*pix_height+line_width)
                overlay_painter.end()
                self.modified = True
                self.update()

        self.dragging = (False, QPoint())

    def __draw_pixel_selector(self, pix_x, pix_y):
        line_width = 4

        # self.overlay.fill(QColorConstants.Transparent)
        overlay_painter = QPainter()
        overlay_painter.begin(self.overlay)
        pen = overlay_painter.pen()
        pen.setWidth(line_width)
        pen.setColor(QColorConstants.Cyan)
        overlay_painter.setPen(pen)
        overlay_painter.drawRect(pix_x*self.scale - int(line_width/2), pix_y*self.scale- int(line_width/2), self.scale+line_width, self.scale+line_width)
        overlay_painter.end()

    def __pixmap(self) -> QPixmap:
        return QPixmap.fromImage(self.background).scaled(self.size())

    def __update_pixmap(self):
        self.pixmap_size = self.size() # QSize(self.width() - 4, self.height() - 4)
        pixmap = QPixmap(self.pixmap_size)
        painter = QPainter()
        painter.begin(pixmap)
        painter.drawPixmap(0, 0, self.__pixmap())
        painter.drawPixmap(0, 0, self.overlay)
        painter.end()
        self.setPixmap(pixmap)

    def as_image(self) -> QPixmap:
        return self.__pixmap()


class ImageEditorWidget(QWidget):

    resized: Signal = Signal(QResizeEvent)
    canvas: ImageCanvas

    def __init__(self, parent=None, init_img: QPixmap = None):
        """
            Creates GUI for image manipulator. Takes an image as input
            for initializing the canvas.
        """
        super(ImageEditorWidget, self).__init__()
        #widgets
        if init_img is None: 
            img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test.png")
            self.canvas = ImageCanvas(img_path=img_path)
        else:
            self.canvas = ImageCanvas(init_img=init_img)

        self.label = QLabel(text="When selected, drawing while dragging is enabled")

        self.checkbox_pen = QCheckBox("Enable Pen Mode")
        self.checkbox_pen.stateChanged.connect(lambda _: self.canvas.set_pen_mode(self.checkbox_pen.isChecked()))

        self.scale_label = QLabel(f"scale:{self.canvas.scale}")
        self.scale_spinner = QSpinBox()
        self.scale_spinner.setRange(10, 16)
        self.update_scale_button = QPushButton("update scale")
        self.update_scale_button.clicked.connect(lambda _: self.change_scale(self.scale_spinner.value()))
        self.scale_spinner.setValue(self.canvas.scale)

        self.color_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_slider.setStyleSheet('''
        QSlider::groove:horizontal {
            height: 15px;
            /* background: #021017;*/
            background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0);
            border-radius: 5px;
            left: 10px;
            right: 10px;
            margin: 2px;
        }
        QSlider::handle:horizontal {
            width: 20px;
            background: #FF7276;
            border: 2px solid #777777;
            margin: -2px 0px;
            /* expand outside the groove */
            border-radius: 5px;
        }
        ''')
        self.color_slider.setRange(0, 255)
        self.color_slider.setValue(0)
        self.color_slider.valueChanged.connect(self.update_pen_color)


        #central widget
        self.grid = QGridLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.grid)

        self.submit_btn = QPushButton("Submit")

        self.grid.addWidget(self.canvas, 0, 0, 1, 3)
        self.grid.addWidget(self.checkbox_pen, 1, 0)
        self.grid.addWidget(self.label, 1, 1)
        self.grid.addWidget(self.scale_spinner, 2, 0)
        self.grid.addWidget(self.update_scale_button, 2, 1)
        self.grid.addWidget(self.scale_label, 2, 2) 
        self.grid.addWidget(self.color_slider, 3, 0, 1, 3)
        self.grid.addWidget(self.submit_btn, 4, 0, 1, 3)

        self.fix_size()

        self.setLayout(self.grid)

    def fix_size(self):
        self.setFixedWidth(self.grid.sizeHint().width())
        self.setFixedHeight(self.grid.sizeHint().height()) 

    def change_scale(self, sc):
        """
            Updates the scale of the canvas.
        """
        self.canvas.update_scale(sc)
        self.scale_label.setText(f"scale:{self.canvas.scale}")
        self.fix_size()

    def update_pen_color(self, c):
        """
            Updates pen color.
        """
        self.canvas.update_color(c)

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.resized.emit(a0)

class ImageEditor(QMainWindow):

    def __init__(self, title: str, mainscreen: QScreen, enable_fileload: bool = False):
        super(ImageEditor, self).__init__(parent=None)

        self.setWindowTitle(title)

        #menubar setup
        menubar = self.menuBar()
        exit_action = QAction(u"&Exit", self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(lambda f: sys.exit(0))

        #loading image file
        open_action = QAction(u"Load File", self)
        open_action.setStatusTip("Load image file")
        open_action.triggered.connect(lambda : self.load_image(self.get_image_file()))

        save_action = QAction(u"Save As Image", self)
        save_action.triggered.connect(lambda : self.save_canvas_as_image(self.image_editor.canvas.as_image()))

        file_menu = menubar.addMenu("&File")    
        file_menu.addAction(exit_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        #central image editor widget
        self.image_editor = ImageEditorWidget()
        self.setCentralWidget(self.image_editor)
        self.image_editor.resized.connect(lambda s: self.setFixedSize(self.__fixed_size_from_widget_size(s.size())))

    def load_image(self, path: str):
        ###use imageeditor's canvas
        self.image_editor.canvas.update_background(path)

    def get_image_file(self) -> str:
        return QFileDialog.getOpenFileName(self, "Open Image", "","Image files (*.jpg *.png)")[0]
    
    def save_canvas_as_image(self, img: QPixmap):
        filename = QFileDialog.getSaveFileName(self, "Save Image", "", "Image files (*.jpg *.png)")[0]
        print(f"Save as: {filename}")
        img.save(filename)

    def __fixed_size_from_widget_size(self, s: QSize) -> QSize:
        return QSize(s.width(), s.height() + self.menuBar().sizeHint().height())


class ImageEditorDialog(QDialog):

    def __init__(self, parent = None, init_img: QPixmap = None):
        super(ImageEditorDialog, self).__init__(parent)

        self.setWindowTitle("Image Editor")

        #main widget is ofc the imageeditor widget
        self.image_editor = ImageEditorWidget(parent, init_img)

        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0) #removes unnecessary margins
        self.layout.addWidget(self.image_editor, alignment=Qt.AlignmentFlag.AlignCenter)
        self.setLayout(self.layout)
        self.image_editor.resized.connect(lambda s: self.setFixedSize(s.size()))
        self.image_editor.submit_btn.clicked.connect(self.accept)

    @staticmethod
    def get_edited_image(parent = None, init_img: QPixmap = None) -> QPixmap:
        """
            Opens the ImageEditorWidget as a dialog and returns the altered image.
        """
        dialog = ImageEditorDialog(parent, init_img)
        dialog.exec()
        return dialog.image_editor.canvas.as_image()
    



if  __name__ == "__main__":
    dialog = True
    app = QApplication([])
    if dialog:
        dialog = ImageEditorDialog()
        img = dialog.get_edited_image()
        img.save("yoo.png")
    else:
        window = ImageEditor("ImageEditor", app.screens()[-1])
        window.show()
        app.exec()
