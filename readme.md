### 环境
pytorch1.6

### 训练步骤
- 将train.py中的data_txt和test_txt改为自己的图片路径文件，每一行是图片的绝对路径，同时label文件和图片在同一文件夹，label文件是coco形式
- python train.py
- 训练好的模型在result/model

### 预测
- 改为自己的模型`net.load_darknet('/home/lwd/code/darknet/backup/yolov3-tiny_best.weights')`和图片路径文件`for line in open('/home/lwd/data/20220523.txt'):`
- python predict.py
- 结果在result/image
