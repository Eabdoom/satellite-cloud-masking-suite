
import os
import sys
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QMessageBox, QSlider, QRadioButton, QButtonGroup
)

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else "intern5"
DATASET_ROOT = Path(r"C:\Users\arnav\Personal\Internships\XDLINX Space Labs\interns_dataset") / DATASET_NAME
IMAGES_DIR = DATASET_ROOT / "images"
MASKS_DIR = DATASET_ROOT / "masks"
PROGRESS_FILE = DATASET_ROOT / "progress.txt"


class DrawLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.main_window = None
        self.setContextMenuPolicy(Qt.PreventContextMenu)

    def mousePressEvent(self, event):
        if self.main_window:
            self.main_window.save_undo_state()
            erase = (event.button() == Qt.RightButton)
            self.main_window.paint_on_mask(event.pos(), self, erase=erase)

    def mouseMoveEvent(self, event):
        if self.main_window:
            if event.buttons() & Qt.LeftButton:
                self.main_window.paint_on_mask(event.pos(), self, erase=False)
            elif event.buttons() & Qt.RightButton:
                self.main_window.paint_on_mask(event.pos(), self, erase=True)


class CloudAnnotator(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Thick Cloud Annotation Tool")
        self.resize(1900, 850)

        self.brush_size = 15
        self.redraw_mode = False
        self.current_threshold = 95

        self.undo_mask = None
        self.image_array = None
        self.mask_array = None

        self.image_files = sorted([
            f for f in os.listdir(IMAGES_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not self.image_files:
            raise RuntimeError("No images found in images folder")

        self.current_index = self.load_progress()

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 5px; color: #1e1e1e;")

        self.original_label = DrawLabel()
        self.mask_label = DrawLabel()
        self.overlay_label = DrawLabel()

        # Center images within their layout columns
        self.original_label.setAlignment(Qt.AlignCenter)
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setAlignment(Qt.AlignCenter)

        self.original_label.main_window = self
        self.mask_label.main_window = self
        self.overlay_label.main_window = self

        image_row = QHBoxLayout()
        image_row.addWidget(self.original_label)
        image_row.addWidget(self.mask_label)
        image_row.addWidget(self.overlay_label)

        # Sliders Layout
        sliders_layout = QHBoxLayout()
        
        # Threshold Slider Setup
        threshold_lbl = QLabel(f"Threshold: {self.current_threshold}")
        threshold_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e1e1e; min-width: 130px;")
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setValue(self.current_threshold)
        self.slider.setFocusPolicy(Qt.NoFocus)
        self.slider.valueChanged.connect(self.threshold_changed)
        
        # Opacity Slider Setup
        self.mask_opacity = 0.5
        opacity_lbl = QLabel("Opacity: 50%")
        opacity_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #1e1e1e; min-width: 130px; margin-left: 20px;")
        
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(50)
        self.opacity_slider.setFocusPolicy(Qt.NoFocus)
        self.opacity_slider.valueChanged.connect(self.opacity_changed)
        
        # Premium Modern Slider styling (apply to both)
        slider_style = """
            QSlider::groove:horizontal {
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                width: 18px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #0056b3;
            }
        """
        self.slider.setStyleSheet(slider_style)
        self.opacity_slider.setStyleSheet(slider_style)
        
        sliders_layout.addWidget(threshold_lbl)
        sliders_layout.addWidget(self.slider)
        sliders_layout.addWidget(opacity_lbl)
        sliders_layout.addWidget(self.opacity_slider)
        
        self.threshold_label = threshold_lbl
        self.opacity_label = opacity_lbl

        # Legend Setup
        legend_lbl = QLabel(
            "<b>Controls Legend:</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Mouse:</b> Left-Click = Draw, Right-Click = Erase &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Brush Size:</b> [ Shrink, ] Grow &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Navigation:</b> Left/Right = Prev/Next &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Threshold:</b> Up/Down = Adjust &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Re-Mask:</b> M = Current Image &nbsp;&nbsp;|&nbsp;&nbsp; "
            "<b>Shortcuts:</b> Ctrl+Z=Undo, R=Clear, D=Delete"
        )
        legend_lbl.setStyleSheet("font-size: 14px; color: #333333; background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 5px;")

        # --- Algorithm Selector Row ---
        algo_layout = QHBoxLayout()

        algo_lbl = QLabel("Re-Mask Algorithm:")
        algo_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #1e1e1e; margin-right: 10px;")

        self.algo_group = QButtonGroup(self)
        radio_style = "font-size: 15px; padding: 4px 12px;"

        self.radio_fixed  = QRadioButton("Fixed Threshold")
        self.radio_otsu   = QRadioButton("Otsu")
        self.radio_hsv    = QRadioButton("HSV")
        self.radio_kmeans = QRadioButton("K-Means")
        self.radio_hsv.setChecked(True)  # HSV is default

        for r in [self.radio_fixed, self.radio_otsu, self.radio_hsv, self.radio_kmeans]:
            r.setStyleSheet(radio_style)
            r.setFocusPolicy(Qt.NoFocus)
            self.algo_group.addButton(r)

        remask_btn = QPushButton("Re-Mask (M)")
        remask_btn.setFocusPolicy(Qt.NoFocus)
        remask_btn.clicked.connect(self.remask_current)

        remask_all_btn = QPushButton("Remask All Remaining")
        remask_all_btn.setFocusPolicy(Qt.NoFocus)
        remask_all_btn.clicked.connect(self.remask_all_remaining)

        remask_btn_style = """
            QPushButton {
                font-size: 15px; font-weight: bold;
                padding: 8px 16px; border-radius: 5px;
                background-color: #17a2b8; color: white;
                border: none;
            }
            QPushButton:hover { background-color: #138496; }
            QPushButton:pressed { background-color: #117a8b; }
        """
        remask_all_style = """
            QPushButton {
                font-size: 15px; font-weight: bold;
                padding: 8px 16px; border-radius: 5px;
                background-color: #fd7e14; color: white;
                border: none;
            }
            QPushButton:hover { background-color: #e8710a; }
            QPushButton:pressed { background-color: #d4650a; }
        """
        remask_btn.setStyleSheet(remask_btn_style)
        remask_all_btn.setStyleSheet(remask_all_style)

        algo_layout.addWidget(algo_lbl)
        algo_layout.addWidget(self.radio_fixed)
        algo_layout.addWidget(self.radio_otsu)
        algo_layout.addWidget(self.radio_hsv)
        algo_layout.addWidget(self.radio_kmeans)
        algo_layout.addSpacing(20)
        algo_layout.addWidget(remask_btn)
        algo_layout.addWidget(remask_all_btn)
        algo_layout.addStretch()

        controls = QHBoxLayout()

        redraw_btn = QPushButton("R - Clear Mask")
        redraw_btn.setFocusPolicy(Qt.NoFocus)
        redraw_btn.clicked.connect(self.start_redraw)

        prev_btn = QPushButton("Previous")
        prev_btn.setFocusPolicy(Qt.NoFocus)
        prev_btn.clicked.connect(self.previous_image)

        next_btn = QPushButton("Next")
        next_btn.setFocusPolicy(Qt.NoFocus)
        next_btn.clicked.connect(self.next_image)

        delete_btn = QPushButton("Delete Pair")
        delete_btn.setFocusPolicy(Qt.NoFocus)
        delete_btn.clicked.connect(self.delete_pair)

        # Premium modern button styling
        btn_style = """
            QPushButton {
                font-size: 16px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                color: #212529;
            }
            QPushButton:hover {
                background-color: #e2e6ea;
                border-color: #dae0e5;
            }
            QPushButton:pressed {
                background-color: #dae0e5;
            }
        """
        redraw_btn.setStyleSheet(btn_style)
        prev_btn.setStyleSheet(btn_style)
        next_btn.setStyleSheet(btn_style)
        delete_btn.setStyleSheet(btn_style)

        controls.addWidget(redraw_btn)
        controls.addWidget(prev_btn)
        controls.addWidget(next_btn)
        controls.addWidget(delete_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(image_row)
        layout.addLayout(sliders_layout)
        layout.addLayout(algo_layout)
        layout.addWidget(legend_lbl)
        layout.addLayout(controls)

        self.setLayout(layout)

        self.load_current_image()

    def load_progress(self):
        if PROGRESS_FILE.exists():
            try:
                idx = int(PROGRESS_FILE.read_text().strip())
                return min(idx, max(0, len(self.image_files) - 1))
            except:
                pass
        return 0

    def save_progress(self):
        PROGRESS_FILE.write_text(str(self.current_index))

    def get_current_paths(self):
        image_name = self.image_files[self.current_index]
        stem = Path(image_name).stem

        image_path = IMAGES_DIR / image_name
        mask_path = MASKS_DIR / f"{stem}_mask.png"

        return image_path, mask_path

    def load_current_image(self):

        image_path, mask_path = self.get_current_paths()

        self.image_array = np.array(
            Image.open(image_path).convert("RGB")
        )

        h, w = self.image_array.shape[:2]

        if mask_path.exists():
            self.mask_array = np.array(
                Image.open(mask_path).convert("L")
            )
        else:
            # Pre-generate threshold mask using current threshold value
            gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, self.current_threshold, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            self.mask_array = mask

        self.redraw_mode = False
        self.update_views()
        self.update_custom_cursor()

    def update_views(self):
        if self.image_array is None:
            return

        h, w = self.image_array.shape[:2]
        display_size = 600

        img_q = QImage(
            self.image_array.data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888
        )

        self.original_label.setPixmap(
            QPixmap.fromImage(img_q).scaled(
                display_size,
                display_size,
                Qt.KeepAspectRatio
            )
        )

        mask_rgb = np.stack([self.mask_array] * 3, axis=-1)

        mask_q = QImage(
            mask_rgb.data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888
        )

        self.mask_label.setPixmap(
            QPixmap.fromImage(mask_q).scaled(
                display_size,
                display_size,
                Qt.KeepAspectRatio
            )
        )

        overlay = self.image_array.copy()
        mask_indices = self.mask_array > 0
        alpha = self.mask_opacity
        overlay[mask_indices] = ((1 - alpha) * overlay[mask_indices] + alpha * np.array([255, 0, 0])).astype(np.uint8)

        overlay_q = QImage(
            overlay.data,
            w,
            h,
            3 * w,
            QImage.Format_RGB888
        )

        self.overlay_label.setPixmap(
            QPixmap.fromImage(overlay_q).scaled(
                display_size,
                display_size,
                Qt.KeepAspectRatio
            )
        )

        self.status_label.setText(
            f"Image {self.current_index + 1}/{len(self.image_files)} | "
            f"{self.image_files[self.current_index]} | "
            f"Brush={self.brush_size} | "
            f"Threshold={self.current_threshold} | "
            f"Controls: Left-Click = Draw, Right-Click = Erase"
        )

    def update_custom_cursor(self):
        if self.image_array is None:
            return
        img_h, img_w = self.image_array.shape[:2]
        
        display_size = 600
        aspect = img_w / img_h
        if aspect >= 1.0:
            label_w = display_size
            label_h = int(display_size / aspect)
        else:
            label_h = display_size
            label_w = int(display_size * aspect)

        radius = int(self.brush_size * label_w / img_w)
        radius = max(1, radius)
        diameter = radius * 2
        
        size = diameter + 4
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(QPen(QColor(255, 255, 255, 255), 1.5, Qt.SolidLine))
        painter.drawEllipse(2, 2, diameter, diameter)
        
        painter.setPen(QPen(QColor(0, 255, 255, 255), 1.0, Qt.SolidLine))
        painter.setBrush(QColor(0, 255, 255, 60))
        painter.drawEllipse(2, 2, diameter, diameter)
        
        painter.end()
        
        cursor = QCursor(pixmap, size // 2, size // 2)
        self.original_label.setCursor(cursor)
        self.mask_label.setCursor(cursor)
        self.overlay_label.setCursor(cursor)

    def threshold_changed(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
        self.current_threshold = value
        self.apply_threshold()

    def opacity_changed(self, value):
        self.mask_opacity = value / 100.0
        self.opacity_label.setText(f"Opacity: {value}%")
        self.update_views()

    def apply_threshold(self):
        if self.image_array is None:
            return
        
        gray = cv2.cvtColor(self.image_array, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.current_threshold, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        self.save_undo_state()
        self.mask_array = mask
        self.update_views()

    def get_selected_algo(self):
        if self.radio_fixed.isChecked():
            return "fixed"
        elif self.radio_otsu.isChecked():
            return "otsu"
        elif self.radio_hsv.isChecked():
            return "hsv"
        else:
            return "kmeans"

    def generate_mask_array(self, img_rgb, mode):
        """Generate a binary mask numpy array from an RGB image using the given mode."""
        kernel = np.ones((5, 5), np.uint8)

        if mode == "hsv":
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            _, s, v = cv2.split(hsv)
            _, low_sat  = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY_INV)
            _, high_val = cv2.threshold(v, 100, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(low_sat, high_val)

        elif mode == "otsu":
            gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif mode == "kmeans":
            img_bgr    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            pixel_data = img_bgr.reshape((-1, 3)).astype(np.float32)
            criteria   = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixel_data, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            cloud_idx  = int(np.argmax(centers.mean(axis=1)))
            flat       = np.where(labels.flatten() == cloud_idx, 255, 0).astype(np.uint8)
            mask       = flat.reshape(img_rgb.shape[:2])

        else:  # fixed
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray, self.current_threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def remask_current(self):
        """Re-generate the mask for the current image using the selected algorithm."""
        if self.image_array is None:
            return
        algo = self.get_selected_algo()
        self.status_label.setText(f"Running {algo.upper()} on current image...")
        QApplication.processEvents()
        self.save_undo_state()
        self.mask_array = self.generate_mask_array(self.image_array, algo)
        self.update_views()

    def remask_all_remaining(self):
        """Bulk re-generate masks for all images from current_index to end."""
        total_remaining = len(self.image_files) - self.current_index
        algo = self.get_selected_algo()

        answer = QMessageBox.question(
            self,
            "Remask All Remaining",
            f"Overwrite masks for {total_remaining} images\n"
            f"(from image {self.current_index + 1} to {len(self.image_files)})\n"
            f"using algorithm: {algo.upper()}?\n\n"
            f"This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if answer != QMessageBox.Yes:
            return

        # Save current mask before bulk operation
        self.save_mask()

        for i in range(self.current_index, len(self.image_files)):
            img_name  = self.image_files[i]
            img_path  = IMAGES_DIR / img_name
            stem      = Path(img_name).stem
            mask_path = MASKS_DIR / f"{stem}_mask.png"

            self.status_label.setText(
                f"Remasking [{i + 1}/{len(self.image_files)}] {img_name} using {algo.upper()}..."
            )
            QApplication.processEvents()

            img_arr = np.array(Image.open(img_path).convert("RGB"))
            mask    = self.generate_mask_array(img_arr, algo)
            Image.fromarray(mask).save(mask_path)

        # Reload current image to show fresh mask
        self.load_current_image()
        self.status_label.setText(
            f"Done! Remasked {total_remaining} images with {algo.upper()}."
        )

    def save_undo_state(self):
        self.undo_mask = self.mask_array.copy()

    def start_redraw(self):
        self.save_undo_state()
        self.mask_array[:] = 0
        self.update_views()

    def paint_on_mask(self, pos, label, erase=False):
        pixmap = label.pixmap()
        if pixmap is None:
            return

        img_h, img_w = self.mask_array.shape

        label_w = label.width()
        label_h = label.height()

        x = int(pos.x() * img_w / max(1, label_w))
        y = int(pos.y() * img_h / max(1, label_h))

        yy, xx = np.ogrid[:img_h, :img_w]

        circle = (xx - x) ** 2 + (yy - y) ** 2 <= self.brush_size ** 2

        if erase:
            self.mask_array[circle] = 0
        else:
            self.mask_array[circle] = 255

        self.update_views()

    def save_mask(self):

        _, mask_path = self.get_current_paths()

        Image.fromarray(self.mask_array).save(mask_path)

    def next_image(self):

        self.save_mask()

        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1

        self.save_progress()

        self.load_current_image()

    def previous_image(self):

        self.save_mask()

        if self.current_index > 0:
            self.current_index -= 1

        self.save_progress()

        self.load_current_image()

    def delete_pair(self):

        image_path, mask_path = self.get_current_paths()

        answer = QMessageBox.question(
            self,
            "Delete",
            f"Delete?\n\n{image_path.name}\n{mask_path.name}"
        )

        if answer != QMessageBox.Yes:
            return

        if image_path.exists():
            os.remove(image_path)

        if mask_path.exists():
            os.remove(mask_path)

        self.image_files.pop(self.current_index)

        if not self.image_files:
            QMessageBox.information(self, "Done", "No images left.")
            self.close()
            return

        self.current_index = min(
            self.current_index,
            len(self.image_files) - 1
        )

        self.save_progress()

        self.load_current_image()

    def undo(self):

        if self.undo_mask is not None:
            self.mask_array = self.undo_mask.copy()
            self.update_views()

    def keyPressEvent(self, event):

        key = event.key()

        if key == Qt.Key_R:
            self.start_redraw()

        elif key == Qt.Key_Right:
            self.next_image()

        elif key == Qt.Key_Left:
            self.previous_image()

        elif key == Qt.Key_D:
            self.delete_pair()

        elif key == Qt.Key_BracketRight:
            self.brush_size = min(100, self.brush_size + 2)
            self.update_views()
            self.update_custom_cursor()

        elif key == Qt.Key_BracketLeft:
            self.brush_size = max(1, self.brush_size - 2)
            self.update_views()
            self.update_custom_cursor()

        elif key == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo()

        elif key == Qt.Key_Up:
            self.current_threshold = min(255, self.current_threshold + 5)
            self.slider.setValue(self.current_threshold)

        elif key == Qt.Key_Down:
            self.current_threshold = max(0, self.current_threshold - 5)
            self.slider.setValue(self.current_threshold)

        elif key == Qt.Key_M:
            self.remask_current()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    win = CloudAnnotator()
    win.show()

    sys.exit(app.exec_())




