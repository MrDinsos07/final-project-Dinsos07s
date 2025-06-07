import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from DataPage import DataCapturePage
from TrainPage import TrainPage
from TestPage import TestPage

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification Tool")
        self.resize(1000, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.data_page = DataCapturePage()
        self.train_page = TrainPage(log_file="train_log.json")
        self.test_page = TestPage()

        self.tabs.addTab(self.data_page, "📸 เก็บข้อมูล")
        self.tabs.addTab(self.train_page, "🧠 ฝึกโมเดล")
        self.tabs.addTab(self.test_page, "🔍 ทดสอบโมเดล")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainApp()
    main_win.show()
    sys.exit(app.exec_())
