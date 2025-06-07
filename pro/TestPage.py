from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

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
