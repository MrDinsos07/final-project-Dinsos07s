from PyQt5.QtWidgets import QWidget
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QLineEdit, QSpinBox, QVBoxLayout, QHBoxLayout, QComboBox, QFileDialog, QCheckBox, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
import os
from time import time

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
