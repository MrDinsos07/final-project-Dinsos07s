from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QScrollArea, QGridLayout, QHBoxLayout, QSpinBox, QProgressBar, QTextEdit, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import os
import subprocess
import sys
import json

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

        self.export_js_btn = QPushButton("แปลงเป็น JavaScript")
        self.export_js_btn.clicked.connect(self.export_to_js)
        layout.addWidget(self.export_js_btn)

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
    
    def export_to_js(self):
        if not os.path.exists("model.keras"):
            QMessageBox.warning(self, "ข้อผิดพลาด", "ยังไม่มีโมเดลให้แปลง")
            return

        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์เพื่อบันทึกโมเดล JavaScript")
        if folder:
            try:
                # เรียกคำสั่ง tensorflowjs_converter
                command = [
                    "tensorflowjs_converter",
                    "--input_format=keras",
                    "model.keras",
                    folder
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode == 0:
                    self.log_output.append(f"แปลงโมเดลเป็น JavaScript สำเร็จใน: {folder}")
                else:
                    self.log_output.append(f"เกิดข้อผิดพลาดในการแปลง:\n{result.stderr}")
            except Exception as e:
                QMessageBox.critical(self, "ข้อผิดพลาด", f"ไม่สามารถแปลงเป็น JavaScript ได้: {e}")
