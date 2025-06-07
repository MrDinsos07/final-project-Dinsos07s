import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QPushButton,
    QFileDialog, QLabel, QSpinBox, QTextEdit, QHBoxLayout, QGridLayout,
    QScrollArea, QMessageBox, QProgressBar, QComboBox, QCheckBox, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import subprocess
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Train Page
class TrainPage(QWidget):
    def __init__(self, log_file):
        super().__init__()
        self.dataset_path = ""
        self.train_process = None
        self.log_file = log_file

        layout = QVBoxLayout()

        self.choose_btn = QPushButton("เลือกโฟลเดอร์ข้อมูล")
        self.choose_btn.clicked.connect(self.choose_dataset)
        layout.addWidget(self.choose_btn)

        self.dataset_label = QLabel("ยังไม่ได้เลือกโฟลเดอร์")
        layout.addWidget(self.dataset_label)

        self.preview_area = QScrollArea()
        self.preview_widget = QWidget()
        self.preview_layout = QGridLayout()
        self.preview_widget.setLayout(self.preview_layout)
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setWidget(self.preview_widget)
        layout.addWidget(QLabel("ตัวอย่างข้อมูล"))
        layout.addWidget(self.preview_area)

        self.status_label = QLabel("สถานะ: ยังไม่ได้เริ่มฝึก")
        layout.addWidget(self.status_label)

        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("จำนวนรอบ:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 100)
        self.epoch_spin.setValue(15)
        h_layout.addWidget(self.epoch_spin)
        layout.addLayout(h_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self.epoch_spin.value())
        layout.addWidget(self.progress_bar)

        self.start_btn = QPushButton("เริ่มฝึก")
        self.start_btn.clicked.connect(self.start_training)
        layout.addWidget(self.start_btn)

        self.save_btn = QPushButton("บันทึกโมเดล")
        self.save_btn.clicked.connect(self.save_model)
        layout.addWidget(self.save_btn)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_log)
        self.timer.start(1000)

    def choose_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ข้อมูล")
        if path:
            self.dataset_path = path
            self.dataset_label.setText(f"ชุดข้อมูล: {path}")
            self.load_preview()

    def load_preview(self):
        for i in reversed(range(self.preview_layout.count())):
            widget = self.preview_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        row, col = 0, 0
        max_preview = 12
        count = 0

        for class_name in os.listdir(self.dataset_path):
            class_dir = os.path.join(self.dataset_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir)[:2]:
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img = img.resize((80, 80))
                    qt_img = QImage(img.tobytes(), img.width, img.height, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_img)

                    label = QLabel(f"{class_name}")
                    label.setAlignment(Qt.AlignCenter)

                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    img_label.setAlignment(Qt.AlignCenter)

                    vbox = QVBoxLayout()
                    vbox.addWidget(img_label)
                    vbox.addWidget(label)

                    container = QWidget()
                    container.setLayout(vbox)
                    self.preview_layout.addWidget(container, row, col)

                    col += 1
                    if col >= 4:
                        row += 1
                        col = 0

                    count += 1
                    if count >= max_preview:
                        return
                except Exception as e:
                    print(f"Error loading image: {e}")

    def start_training(self):
        if not self.dataset_path:
            self.log_output.append("กรุณาเลือกโฟลเดอร์ข้อมูลก่อน")
            return

        epochs = self.epoch_spin.value()
        self.progress_bar.setMaximum(epochs)
        self.progress_bar.setValue(0)

        command = [
            sys.executable, 'train_model.py',
            self.dataset_path,
            str(epochs),
            self.log_file
        ]

        self.log_output.append(f"เริ่มฝึกด้วย {epochs} รอบ...")
        self.status_label.setText("สถานะ: กำลังฝึก...")
        self.train_process = subprocess.Popen(command)
        self.start_btn.setEnabled(False)

    def update_log(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file) as f:
                    data = json.load(f)
                    output = (
                        f"รอบที่ {data['epoch']}\n"
                        f"  - [Train] acc: {data['accuracy']:.4f}, loss: {data['loss']:.4f}\n"
                        f"  - [Val]   acc: {data['val_accuracy']:.4f}, loss: {data['val_loss']:.4f}"
                    )
                    self.log_output.setText(output)
                    self.progress_bar.setValue(data["epoch"])
            except Exception:
                pass

        if self.train_process and self.train_process.poll() is not None:
            self.log_output.append("การฝึกเสร็จสิ้น")
            self.status_label.setText("สถานะ: ฝึกเสร็จแล้ว")
            self.start_btn.setEnabled(True)
            self.train_process = None

    def save_model(self):
        if not os.path.exists("model.keras") or not os.path.exists("label.npy"):
            QMessageBox.warning(self, "ข้อผิดพลาด", "ยังไม่มีโมเดลหรือ label ให้บันทึก")
            return

        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์เพื่อบันทึกโมเดล")
        if folder:
            try:
                import shutil
                shutil.copy("model.keras", os.path.join(folder, "model.keras"))
                shutil.copy("label.npy", os.path.join(folder, "label.npy"))
                self.log_output.append(f"โมเดลบันทึกไว้ที่: {folder}")
            except Exception as e:
                QMessageBox.critical(self, "ข้อผิดพลาด", f"เกิดข้อผิดพลาดในการบันทึก: {e}")

# Data Capture Page
def detect_cameras(max_devices=5):
    available = []
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available

class DataCapturePage(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ระบบจับภาพใบหน้าอัตโนมัติ")

        # UI Elements
        self.folder_name_input = QLineEdit()
        self.base_folder = ""
        self.folder_path_label = QLabel("ยังไม่ได้เลือกตำแหน่งบันทึก")
        self.select_folder_button = QPushButton("📂 เลือกตำแหน่ง...")

        self.num_images_input = QSpinBox()
        self.num_images_input.setRange(1, 1000)
        self.num_images_input.setValue(40)

        self.camera_select = QComboBox()
        self.detect_and_fill_cameras()

        self.show_frame_checkbox = QCheckBox("แสดงกรอบสีแดงรอบใบหน้า")
        self.show_frame_checkbox.setChecked(True)

        self.status_label = QLabel("สถานะ: รอเริ่มถ่ายภาพ")
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 400)

        self.start_button = QPushButton("▶ เริ่มถ่ายภาพ")
        self.cancel_button = QPushButton("🛑 ยกเลิก")
        self.cancel_button.setEnabled(False)

        # Layouts
        layout = QVBoxLayout()
        form_layout = QVBoxLayout()

        folder_path_layout = QHBoxLayout()
        folder_path_layout.addWidget(QLabel("📂 ตำแหน่งบันทึก:"))
        folder_path_layout.addWidget(self.folder_path_label)
        folder_path_layout.addWidget(self.select_folder_button)

        name_input_layout = QHBoxLayout()
        name_input_layout.addWidget(QLabel("🗂️ ชื่อโฟลเดอร์ย่อย:"))
        name_input_layout.addWidget(self.folder_name_input)
        name_input_layout.addWidget(QLabel("📸 จำนวนภาพ:"))
        name_input_layout.addWidget(self.num_images_input)
        name_input_layout.addWidget(QLabel("🎥 เลือกกล้อง:"))
        name_input_layout.addWidget(self.camera_select)

        form_layout.addLayout(folder_path_layout)
        form_layout.addLayout(name_input_layout)
        form_layout.addWidget(self.show_frame_checkbox)

        layout.addLayout(form_layout)
        layout.addWidget(self.video_label)
        layout.addWidget(self.status_label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        # Logic setup
        self.cap = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_button.clicked.connect(self.start_capture)
        self.cancel_button.clicked.connect(self.stop_capture)
        self.select_folder_button.clicked.connect(self.select_base_folder)
        self.camera_select.currentIndexChanged.connect(self.change_camera)

        self.capturing = False
        self.captured = 0
        self.num_images = 0
        self.output_folder = ""
        self.last_capture_time = 0
        self.countdown_seconds = 3

        self.init_camera()

    def detect_and_fill_cameras(self):
        cameras = detect_cameras()
        if not cameras:
            QMessageBox.critical(self, "ไม่พบกล้อง", "ไม่พบกล้องที่ใช้งานได้")
        self.camera_select.deleteLater()
        for cam_index in cameras:
            self.camera_select.addItem(str(cam_index))

    def init_camera(self):
        if self.cap:
            self.cap.release()
        cam_index = int(self.camera_select.currentText()) if self.camera_select.count() > 0 else 0
        self.cap = cv2.VideoCapture(cam_index)
        if self.cap.isOpened():
            self.timer.start(30)
        else:
            QMessageBox.critical(self, "ข้อผิดพลาด", "ไม่สามารถเปิดกล้องได้")

    def change_camera(self):
        self.init_camera()

    def select_base_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกตำแหน่งบันทึก", os.path.expanduser("~"))
        if folder:
            self.base_folder = folder
            self.folder_path_label.setText(folder)

    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        draw_frame = frame.copy()

        if self.capturing:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            current_time = time()
            if len(faces) > 0:
                (x, y, w, h) = faces[0]

                padding_x = int(w * 0.3)
                padding_y = int(h * 0.5)
                x1 = max(x - padding_x, 0)
                y1 = max(y - padding_y, 0)
                x2 = min(x + w + padding_x, frame.shape[1])
                y2 = min(y + h + padding_y, frame.shape[0])

                if self.show_frame_checkbox.isChecked():
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                if (current_time - self.last_capture_time) >= 0.5:
                    face_img = frame[y1:y2, x1:x2]
                    filename = os.path.join(self.output_folder, f"face_{self.captured + 1:03d}.jpg")
                    cv2.imwrite(filename, face_img)

                    self.captured += 1
                    self.last_capture_time = current_time
                    self.status_label.setText(f"สถานะ: ถ่ายแล้ว {self.captured} / {self.num_images} ภาพ")

                if self.captured >= self.num_images:
                    self.stop_capture()

        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0 and self.show_frame_checkbox.isChecked():
                (x, y, w, h) = faces[0]
                padding_x = int(w * 0.3)
                padding_y = int(h * 0.5)
                x1 = max(x - padding_x, 0)
                y1 = max(y - padding_y, 0)
                x2 = min(x + w + padding_x, frame.shape[1])
                y2 = min(y + h + padding_y, frame.shape[0])
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        rgb_image = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def start_capture(self):
        folder_name = self.folder_name_input.text().strip()
        if not folder_name:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณากรอกชื่อโฟลเดอร์ย่อย")
            return

        if not self.base_folder:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาเลือกตำแหน่งบันทึก")
            return

        self.output_folder = os.path.join(self.base_folder, folder_name)
        os.makedirs(self.output_folder, exist_ok=True)

        self.num_images = self.num_images_input.value()
        self.captured = 0
        self.last_capture_time = 0
        self.countdown_seconds = 3
        self.status_label.setText("สถานะ: เริ่มถ่ายใน 3...")
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

        QTimer.singleShot(1000, self.start_countdown)

    def start_countdown(self):
        self.countdown_seconds -= 1
        if self.countdown_seconds > 0:
            self.status_label.setText(f"สถานะ: เริ่มถ่ายใน {self.countdown_seconds}...")
            QTimer.singleShot(1000, self.start_countdown)
        else:
            self.capturing = True
            self.status_label.setText("สถานะ: เริ่มถ่ายภาพ...")
            self.cancel_button.setEnabled(True)

    def stop_capture(self):
        self.capturing = False
        self.status_label.setText("การถ่ายภาพเสร็จสิ้น" if self.captured > 0 else "ยกเลิกการถ่ายภาพ")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        event.accept()

# Placeholder Test Page (Replace with your actual TestPage code)
class TestPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.model = None
        self.labels = None

        self.image_label = QLabel("ยังไม่มีภาพ")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(300, 300)
        layout.addWidget(self.image_label)

        self.load_model_btn = QPushButton("โหลดโมเดล")
        self.load_model_btn.clicked.connect(self.load_model)
        layout.addWidget(self.load_model_btn)

        self.open_image_btn = QPushButton("เปิดภาพทดสอบ")
        self.open_image_btn.clicked.connect(self.open_image)
        layout.addWidget(self.open_image_btn)

        self.test_btn = QPushButton("ทำนาย")
        self.test_btn.clicked.connect(self.test_model)
        self.test_btn.setEnabled(False)
        layout.addWidget(self.test_btn)

        self.result_label = QLabel("ผลลัพธ์: -")
        layout.addWidget(self.result_label)

        self.setLayout(layout)
        self.image_path = ""

    def load_model(self):
        try:
            self.model = load_model("model.keras")
            self.labels = np.load("label.npy", allow_pickle=True)
            QMessageBox.information(self, "โหลดสำเร็จ", "โหลดโมเดลและ labels เรียบร้อยแล้ว")
            self.test_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "ผิดพลาด", f"โหลดโมเดลไม่สำเร็จ: {e}")

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "เลือกภาพ", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def test_model(self):
        if not self.model or not self.image_path:
            QMessageBox.warning(self, "ข้อผิดพลาด", "กรุณาโหลดโมเดลและเลือกรูปภาพก่อน")
            return

        try:
            img = Image.open(self.image_path).convert("RGB").resize((224, 224))
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            pred = self.model.predict(img_array)
            class_index = np.argmax(pred)
            class_name = self.labels[class_index]

            self.result_label.setText(f"ผลลัพธ์: {class_name} ({pred[0][class_index]:.4f})")
        except Exception as e:
            QMessageBox.critical(self, "ผิดพลาด", f"ไม่สามารถทำนายได้: {e}")


# Main Application
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ระบบจัดการโมเดลการจำแนกภาพ")
        self.setGeometry(100, 100, 800, 600)

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Add pages to tabs
        self.data_capture_page = DataCapturePage()
        self.train_page = TrainPage("training_log.json")
        self.test_page = TestPage()

        self.tabs.addTab(self.data_capture_page, "เก็บข้อมูล")
        self.tabs.addTab(self.train_page, "ฝึกโมเดล")
        self.tabs.addTab(self.test_page, "ทดสอบโมเดล")

    def closeEvent(self, event):
        # Ensure camera is released when closing
        if hasattr(self.data_capture_page, 'cap') and self.data_capture_page.cap:
            self.data_capture_page.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())