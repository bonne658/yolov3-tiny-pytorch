import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.length             = len(self.annotation_lines)
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        # 归一化（除以255），whc转chw
        image       = np.transpose(np.array(image, dtype=np.float32)/255.0, (2, 0, 1))
        # 左上右下形式
        box         = np.array(box, dtype=np.float32)
        
        if len(box) != 0:
            # 转化成比例形式
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            # 转化成中心+宽高形式
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.02, sat=1.5, val=1.5, random=True):
        line    = annotation_line.split()
        # 标签：中心+宽高形式
        label_line = line[0][:-4]+'.txt'
        boxes = []
        for lin in open(label_line):
            t = lin.split()
            boxes.append([t[1],t[2],t[3],t[4],t[0]])
        box = np.array(boxes, dtype=np.float32)
        # 图像
        image   = Image.open(line[0])
        iw, ih  = image.size
        h, w    = input_shape
        if len(box) > 0:
        	# 转化成数字形式
        	box[:, [0,2]] = box[:, [0,2]] * iw
        	box[:, [1,3]] = box[:, [1,3]] * ih
        	# 转化成左上右下形式
        	box[:, 0:2] = box[:, 0:2] - box[:, 2:4] / 2
        	box[:, 2:4] = box[:, 0:2] + box[:, 2:4]
        # 验证
        if not random:
            # 计算图片等比例缩放到输入大小的宽高，可能有一个小于输入尺寸
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            # 嵌入坐标
            dx = (w-nw)//2
            dy = (h-nh)//2
            # 放缩
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            # 嵌入在中间
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            if len(box)>0:
                np.random.shuffle(box)
                # 将标签转换到新图片
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                # 左上不小于0
                box[:, 0:2][box[:, 0:2]<0] = 0
                # 右下不大于宽高
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                # 宽高要大于一个像素
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] 
            # 标签形式：数字，左上右下
            return image_data, box
                
        # 训练
        # 宽高的新比率
        new_ar = w/h * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        # 相对输入尺寸的放缩比例
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        # 放缩
        image = image.resize((nw,nh), Image.BICUBIC)
        # 随机一个嵌入坐标
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        # 嵌入
        new_image.paste(image, (dx, dy))
        image = new_image
        # 翻转图像
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        if len(box)>0:
            np.random.shuffle(box)
            # 将标签转换到新图片
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            # 左上不小于0
            box[:, 0:2][box[:, 0:2]<0] = 0
             # 右下不大于宽高
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            # 宽高要大于一个像素
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        # 标签形式：数字，左上右下
        return image_data, box
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
