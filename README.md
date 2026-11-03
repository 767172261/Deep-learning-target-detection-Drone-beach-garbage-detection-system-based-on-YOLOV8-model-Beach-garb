基于YOLOv8的无人机海滩垃圾检测系统
@[toc]

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e7fd024d191145398b6d15d29f485133.png)
---

### **基于YOLOv8的无人机海滩垃圾检测系统**

| **类别** | **内容** |
|----------|----------|
| **项目名称** | 基于YOLOv8的无人机海滩垃圾检测系统 |
| **数据集规模** | 2,800 张图像 |
| **检测类别** | 6 类 |
| **类别名称** | `glass`（玻璃）、`plastic-bottle`（塑料瓶）、`retort-pouch`（蒸煮袋）、`take-away-containers`（外卖餐盒）、`takeaway-cup`（一次性杯）、`tin-and-aluminum-cans`（金属罐） |
| **数据来源** | 无人机航拍图像（真实海滩场景） |
| **标注格式** | YOLO `.txt` 格式（归一化坐标） |
| **模型状态** | 已训练完成，提供 `.pt` 权重文件，可直接调用 |
| **适用任务** | 海滩垃圾识别、环境监测、智能环保巡检 |

---

### **开发环境与依赖**

| 项目 | 配置要求 |
|------|----------|
| **开发软件** | PyCharm + Anaconda 或 VSCode + Anaconda |
| **Python 版本** | Python 3.9 |
| **核心库** | `ultralytics`（YOLOv8）、`torch==1.9`、`opencv-python`、`PyQt5` |
| **GUI 框架** | PyQt5（用于图形界面开发） |
| **运行平台** | Windows / Linux / macOS（支持主流系统） |

---

### **项目文件结构**

| 文件类型 | 包含内容 |
|----------|----------|
| **程序文件** | `.py` 源代码（主程序、检测逻辑、UI绑定等） |
| **UI界面文件** | `.ui`（Qt Designer设计文件）、`.qrc`（资源文件）、转换后的 `.py` 界面代码 |
| **数据集** | 2,800 张带标签的无人机海滩垃圾图像（含 `images` 和 `labels`） |
| **模型文件** | 训练好的 YOLOv8 模型权重文件（如 `best.pt`） |
| **辅助文件** | 项目运行教程（`.txt`）、效果图（`.jpg`）、图标与资源文件等 |

---

### **系统功能列表**

| 功能编号 | 功能描述 |
|----------|----------|
| ① | 支持选择**单张图片**或整个**图片文件夹**进行识别（支持批量检测） |
| ② | 支持上传并检测**视频文件**，逐帧分析并输出结果 |
| ③ | 支持调用**本地摄像头**或无人机视频流进行实时检测 |
| ④ | 支持**自定义置信度阈值**和 **IOU阈值**（用于NMS处理） |
| ⑤ | 支持选择是否显示**类别标签**、**检测框**以及是否显示**原图** |
| ⑥ | 支持**切换和加载不同检测模型**（如不同训练版本的 `.pt` 文件） |
| ⑦ | 批量检测后可在界面中**逐张查看每张图像的检测结果**（可视化浏览） |
| ⑧ | 支持将检测结果（带框图像、视频、统计信息）**保存到本地** |

---

### **类别定义表**

| 类别编号 | 类别名称（英文） | 类别名称（中文） | 说明 |
|----------|------------------|------------------|------|
| 0 | glass | 玻璃 | 海滩常见玻璃瓶或碎片 |
| 1 | plastic-bottle | 塑料瓶 | PET塑料饮料瓶等 |
| 2 | retort-pouch | 蒸煮袋 | 即食食品包装袋（铝塑复合） |
| 3 | take-away-containers | 外卖餐盒 | 泡沫或塑料材质餐盒 |
| 4 | takeaway-cup | 一次性杯 | 咖啡杯、饮料杯等 |
| 5 | tin-and-aluminum-cans | 金属罐 | 易拉罐、罐头盒等 |

---

### **界面显示信息**

| 显示内容 | 说明 |
|----------|------|
| 目标位置 | 在图像上用边界框标出各类垃圾位置 |
| 目标总数 | 实时显示当前画面中检测到的垃圾总数 |
| 分类统计 | 可显示每类垃圾的数量（可选） |
| 置信度 | 显示每个检测框的置信度分数（0~1） |
| 实时反馈 | 视频或摄像头模式下实现近实时检测与显示 |

---



---

> 📌 **应用场景**：  
> - 海滩清洁机器人导航  
> - 无人机环保巡查系统  
> - 城市环卫AI监管平台  
 


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ebb75752af2b4f2ab1a4bad387312d29.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6734b0546f4d407f85e75c5f7acabbf3.png)
构建基于YOLOv8的无人机海滩垃圾检测系统是一个涉及数据准备、模型训练、UI设计和实际应用的综合性项目。以下是详细的步骤和代码示例，帮助你完成这个项目。

### 一、环境配置

#### 1. 安装依赖库
确保安装了以下Python库：
```bash
conda create -n beach_clean python=3.9
conda activate beach_clean
pip install opencv-python PyQt5 ultralytics torch==1.9 torchvision torchaudio
```

### 二、数据集准备

#### 1. 数据集结构
假设你的数据集已经按照如下结构组织：
```
datasets/
└── beach_litter/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

`data.yaml` 文件内容如下：
```yaml
train: ../datasets/beach_litter/images/train
val: ../datasets/beach_litter/images/val
test: ../datasets/beach_litter/images/test

nc: 6
names: ['glass', 'plastic-bottle', 'retort-pouch', 'take-away-containers', 'takeaway-cup', 'tin-and-aluminum-cans']
```

### 三、模型训练

#### 1. 训练脚本 (`train.py`)
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(
    data='datasets/beach_litter/data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    name='beach_litter_yolov8n_640',
    device='cuda',
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,
    project='runs/beach_litter',
    amp=True,
    close_mosaic=30,
    cache='disk'
)
```

#### 2. 准备数据集的代码 (`prepare_data.py`)
```python
import os
import shutil
import random

def prepare_data(src_folder, dst_folder, split_ratio=(0.8, 0.1, 0.1)):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 创建子文件夹
    for folder in ['images', 'labels']:
        for subfolder in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dst_folder, folder, subfolder), exist_ok=True)

    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    total_images = len(image_files)
    train_split = int(total_images * split_ratio[0])
    val_split = int(total_images * (split_ratio[0] + split_ratio[1]))

    for i, image_file in enumerate(image_files):
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if i < train_split:
            dest_image_folder = os.path.join(dst_folder, 'images', 'train')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'train')
        elif i < val_split:
            dest_image_folder = os.path.join(dst_folder, 'images', 'val')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'val')
        else:
            dest_image_folder = os.path.join(dst_folder, 'images', 'test')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'test')

        shutil.copy(os.path.join(src_folder, image_file), dest_image_folder)
        shutil.copy(os.path.join(src_folder, label_file), dest_label_folder)

if __name__ == "__main__":
    src_folder = 'path/to/source/folder'
    dst_folder = 'datasets/beach_litter'
    prepare_data(src_folder, dst_folder)
```

### 四、UI界面开发

#### 1. UI设计 (`ui_main_window.ui`)
使用Qt Designer设计UI界面，并转换为Python代码（`ui_main_window.py`）。

#### 2. 主程序 (`MainProgram.py`)
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_main_window import Ui_MainWindow
from detect_tools import Detector

class BeachLitterDetectionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.detector = Detector()

        # 连接按钮事件
        self.btnSelectImage.clicked.connect(self.select_image)
        self.btnSelectVideo.clicked.connect(self.select_video)
        self.btnSelectCamera.clicked.connect(self.select_camera)
        self.btnSaveResults.clicked.connect(self.save_results)

    def select_image(self):
        file_path = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg)")
        if file_path[0]:
            results = self.detector.detect_image(file_path[0])
            self.display_results(results)

    def select_video(self):
        file_path = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path[0]:
            results = self.detector.detect_video(file_path[0])
            self.display_results(results)

    def select_camera(self):
        results = self.detector.detect_camera()
        self.display_results(results)

    def save_results(self):
        self.detector.save_results()

    def display_results(self, results):
        # 显示结果到UI界面
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeachLitterDetectionApp()
    window.show()
    sys.exit(app.exec_())
```

#### 3. 检测工具 (`detect_tools.py`)
```python
from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self):
        self.model = YOLO('models/best.pt')

    def detect_image(self, image_path):
        results = self.model.predict(source=image_path, conf=0.3, iou=0.45)
        return results

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, conf=0.3, iou=0.45)
            # 处理结果
        cap.release()
        return results

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, conf=0.3, iou=0.45)
            # 处理结果
        cap.release()
        return results

    def save_results(self):
        # 保存结果到文件
        pass
```


从数据准备、模型训练、检测工具、UI界面开发等。

### 一、环境配置与依赖安装

#### `installPackages.py`
```python
import subprocess

def install_dependencies():
    dependencies = [
        'opencv-python',
        'PyQt5',
        'ultralytics',
        'torch==1.9',
        'torchvision',
        'torchaudio'
    ]
    
    for dep in dependencies:
        subprocess.run(['pip', 'install', dep])

if __name__ == "__main__":
    install_dependencies()
```

### 二、数据集准备

#### `prepare_data.py`
```python
import os
import shutil
import random

def prepare_data(src_folder, dst_folder, split_ratio=(0.8, 0.1, 0.1)):
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 创建子文件夹
    for folder in ['images', 'labels']:
        for subfolder in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dst_folder, folder, subfolder), exist_ok=True)

    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg') or f.endswith('.png')]
    random.shuffle(image_files)

    total_images = len(image_files)
    train_split = int(total_images * split_ratio[0])
    val_split = int(total_images * (split_ratio[0] + split_ratio[1]))

    for i, image_file in enumerate(image_files):
        label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt')
        if i < train_split:
            dest_image_folder = os.path.join(dst_folder, 'images', 'train')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'train')
        elif i < val_split:
            dest_image_folder = os.path.join(dst_folder, 'images', 'val')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'val')
        else:
            dest_image_folder = os.path.join(dst_folder, 'images', 'test')
            dest_label_folder = os.path.join(dst_folder, 'labels', 'test')

        shutil.copy(os.path.join(src_folder, image_file), dest_image_folder)
        shutil.copy(os.path.join(src_folder, label_file), dest_label_folder)

if __name__ == "__main__":
    src_folder = 'path/to/source/folder'
    dst_folder = 'datasets/beach_litter'
    prepare_data(src_folder, dst_folder)
```

### 三、模型训练

#### `train.py`
```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 开始训练
results = model.train(
    data='datasets/beach_litter/data.yaml',
    epochs=300,
    imgsz=640,
    batch=16,
    name='beach_litter_yolov8n_640',
    device='cuda',
    workers=8,
    optimizer='AdamW',
    lr0=0.001,
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,
    project='runs/beach_litter',
    amp=True,
    close_mosaic=30,
    cache='disk'
)
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/64bfb6ade2be4021a95fad1771c4dc29.png)

### 四、检测工具

#### `detect_tools.py`
```python
from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self):
        self.model = YOLO('models/best.pt')

    def detect_image(self, image_path):
        results = self.model.predict(source=image_path, conf=0.3, iou=0.45)
        return results

    def detect_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, conf=0.3, iou=0.45)
            # 处理结果
        cap.release()
        return results

    def detect_camera(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.predict(source=frame, conf=0.3, iou=0.45)
            # 处理结果
        cap.release()
        return results

    def save_results(self):
        # 保存结果到文件
        pass
```

### 五、UI界面开发

#### `ui_main_window.ui`（使用Qt Designer设计）
转换为Python代码后得到`ui_main_window.py`

#### `MainProgram.py`
```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_main_window import Ui_MainWindow
from detect_tools import Detector

class BeachLitterDetectionApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.detector = Detector()

        # 连接按钮事件
        self.btnSelectImage.clicked.connect(self.select_image)
        self.btnSelectVideo.clicked.connect(self.select_video)
        self.btnSelectCamera.clicked.connect(self.select_camera)
        self.btnSaveResults.clicked.connect(self.save_results)

    def select_image(self):
        file_path = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg)")
        if file_path[0]:
            results = self.detector.detect_image(file_path[0])
            self.display_results(results)

    def select_video(self):
        file_path = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if file_path[0]:
            results = self.detector.detect_video(file_path[0])
            self.display_results(results)

    def select_camera(self):
        results = self.detector.detect_camera()
        self.display_results(results)

    def save_results(self):
        self.detector.save_results()

    def display_results(self, results):
        # 显示结果到UI界面
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeachLitterDetectionApp()
    window.show()
    sys.exit(app.exec_())
```




