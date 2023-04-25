from tiny import Tiny
from loss import Loss
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import numpy as np
import torch, sys, cv2
import torch.optim as optim
from dataloader import YoloDataset, yolo_dataset_collate

def show_batch(image, label):
	for i in range(len(image)):
		im = np.transpose(image[i]*255.0,(1,2,0)).astype('uint8')[:,:,[2,1,0]]
		ih, iw = np.shape(im)[0:2]
		cv2.imshow("im", im)
		cv2.waitKey(0)
		# for lab in label[i]:
		# 	print(lab)

# data
batch_size = 2
data_txt='/home/lwd/data/train.txt'
with open(data_txt) as f:
	train_lines = f.readlines()
train_dataset=YoloDataset(train_lines, (416, 416), True)
train_data = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
test_txt='/home/lwd/data/test.txt'
with open(test_txt) as f:
	test_lines = f.readlines()
test_dataset=YoloDataset(test_lines, (416, 416), False)
test_data = DataLoader(test_dataset, shuffle = False, batch_size = batch_size, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)
train_step = len(train_lines) // batch_size
val_step = len(test_lines) // batch_size
# net
model_path=''
net=Tiny()
net.init()
net.load_darknet('yolov3-tiny.conv.15')
net = net.cuda()

if len(model_path) > 1:
	paras=torch.load(model_path, map_location='cuda')
	net.load_state_dict(paras)
# hyperparameter
anchors = [[44, 43],  [87, 39],  [64,102], [20, 18],  [43, 21],  [28, 34]]
los = Loss((416, 416), anchors, 80)
lr = 1e-4
optimizer = optim.Adam(net.parameters(), lr, weight_decay = 5e-4)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-4)
# iterator
i = 1
lr_cnt = 0
vl_last = 9
for param in net.backbone.parameters():
	param.requires_grad = False
while True:
	net.train()
	# if i % 111 == 0 and lr > 1e-4:
	# 	lr *= 0.1
	# 	for param_group in optimizer.param_groups:
	# 		param_group["lr"] = lr
	if i == 400:
	# 	optimizer = optim.Adam(net.parameters(), 1e-4, weight_decay = 5e-4)
	# 	lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
		for param in net.backbone.parameters():
			param.requires_grad = True
	train_loss = 0
	for bi, (batch_image, batch_label) in enumerate(train_data):
		loss = 0
		number = 0
		#show_batch(batch_image, batch_label)
		batch_image  = torch.from_numpy(batch_image).type(torch.FloatTensor).cuda()
		batch_label = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch_label]
		optimizer.zero_grad()
		outputs = net(batch_image)
		for oi, output in enumerate(outputs):
			loss_item, num_pos = los(oi, output, batch_label)
			loss += loss_item
			number += num_pos
		loss_value = loss / number
		loss_value.backward()
		optimizer.step()
		train_loss += loss_value.item()
	net.eval()
	val_loss = 0
	with torch.no_grad():
		for bi, (batch_image, batch_label) in enumerate(test_data):
			loss = 0
			number = 0
			# show_batch(batch_image, batch_label)
			batch_image  = torch.from_numpy(batch_image).type(torch.FloatTensor).cuda()
			batch_label = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in batch_label]
			optimizer.zero_grad()
			outputs = net(batch_image)
			for oi, output in enumerate(outputs):
				loss_item, num_pos = los(oi, output, batch_label)
				loss += loss_item
				number += num_pos
			loss_value = loss / number
			val_loss += loss_value.item()
	vl=val_loss / val_step
	print('epoch: ', i, ' ------ train_loss:', train_loss / train_step, '   val_loss:', val_loss / val_step)
	print(optimizer.param_groups[0]['lr'])
		
	if vl < vl_last: 
		torch.save(net.state_dict(), 'result/model/'+str(i)+':'+str(vl)[:5]+'.pth')
		vl_last = vl
		#break
	# lr_scheduler.step()
	if i > 999: 
		break
	i += 1
