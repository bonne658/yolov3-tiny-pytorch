from PIL import Image, ImageDraw
import numpy as np
import torch, copy, time, sys, cv2
from tiny import Tiny

class_num = 1
input_shape = (608, 608)
def get_boxes(output, anchors):
	h=output.size(2)
	w=output.size(3)
	output=output.view(3,5+class_num,h,w).permute(0,2,3,1).contiguous()
	# conf
	conf = torch.sigmoid(output[..., 4])
	cl = torch.sigmoid(output[..., 5:])
	clv, cli = torch.max(cl, -1)
	conf = conf * clv
	mask = conf > 0.15
	conf = conf[mask].unsqueeze(-1)
	cli = cli[mask].unsqueeze(-1)
	# grid
	FloatTensor = torch.cuda.FloatTensor if conf.is_cuda else torch.FloatTensor
	grid_h, grid_w = torch.meshgrid(torch.arange(h), torch.arange(w))
	grid_h = grid_h.repeat(3,1,1).type(FloatTensor)
	grid_w = grid_w.repeat(3,1,1).type(FloatTensor)
	tx = (torch.sigmoid(output[..., 0]) + grid_w) / w
	ty = (torch.sigmoid(output[..., 1]) + grid_h) / h
	tx = tx[mask].unsqueeze(-1)
	ty = ty[mask].unsqueeze(-1)
	# anchor
	aw = torch.Tensor(anchors[0::2]).view(3,1).repeat(1,h*w).view(3,h,w).type(FloatTensor)
	ah = torch.Tensor(anchors[1::2]).view(3,1).repeat(1,h*w).view(3,h,w).type(FloatTensor)
	tw = torch.exp(output[..., 2]) * aw
	th = torch.exp(output[..., 3]) * ah
	tw = tw[mask].unsqueeze(-1)
	th = th[mask].unsqueeze(-1)
	return torch.cat([tx, ty, tw, th, cli, conf], -1)
	
def iou(a,b):
	A=len(a)
	B=len(b)
	area1=a[:,2]*a[:,3]
	area1=area1.unsqueeze(1).expand(A,B)
	area2=b[:,2]*b[:,3]
	area2=area2.unsqueeze(0).expand(A,B)
	ba=torch.zeros(a.shape).cuda()
	bb=torch.zeros(b.shape).cuda()
	ba[:,0:2]=a[:,0:2]-a[:,2:]/2.0
	ba[:,2:]=ba[:,0:2]+a[:,2:]
	bb[:,0:2]=b[:,0:2]-b[:,2:]/2.0
	bb[:,2:]=bb[:,0:2]+b[:,2:]
	ba=ba.unsqueeze(1).expand(A,B,4)
	bb=bb.unsqueeze(0).expand(A,B,4)
	lt=torch.max(ba[:,:,0:2], bb[:,:,0:2])
	rb=torch.min(ba[:,:,2:], bb[:,:,2:])
	inter=torch.clamp((rb-lt),min=0)
	inter=inter[:,:,0]*inter[:,:,1]
	return inter/(area1+area2-inter)

def nms(box):
	box = box[torch.argsort(box[:,-1])]
	result=[]
	while len(box) > 0:
		result.append(box[0])
		if len(box) == 1: break
		ious=iou(box[0:1, 0:4], box[1:, 0:4])
		box=box[1:][ious.squeeze(0) < 0.5]
	return torch.stack(result)

def deal(boxes):
	labels = boxes[:, -2].unique()
	result=[]
	for l in labels:
		box = boxes[boxes[:, -2]==l]
		box = nms(box)
		for b in box: 
			result.append(b)
	return torch.stack(result)

#classes=[]
anchors=[[20, 76,  39, 54,  79,69], [32, 22,  26, 39,  46, 32]]
#for line in open('/home/lwd/code/darknet/data/coco.names'):
	#classes.append(line[:-1])
net=Tiny(class_num)
model_path='/home/lwd/csdn/yolov3-tiny-pytorch/result/model/472:3.171.pth'
paras=torch.load(model_path, map_location='cuda')
net.load_state_dict(paras)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.cuda()
net.eval()
with open('log.txt', 'w') as f:
	for line in open('/home/lwd/data/other.txt'):
		print(line[:-1])
		raw = Image.open(line[:-1])
		ih, iw = np.shape(raw)[0:2]
		# inference
		raw = raw.convert('RGB')
		image = raw.resize(input_shape)
		image = np.array(image, dtype='float32') / 255.0
		image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
		with torch.no_grad():
			images = torch.from_numpy(image)
			#images = images[:,[2,1,0],:,:]
			images = images.cuda()
			outputs = net(images)
		
		draw = ImageDraw.Draw(raw)
		thld_boxes=[]
		for i,output in enumerate(outputs):
			# decode output
			boxes = get_boxes(output, anchors[i])
			if len(boxes) == 0: continue
			boxes[:,0] = boxes[:,0] * iw
			boxes[:,1] = boxes[:,1] * ih
			boxes[:,2] = boxes[:,2] / float(input_shape[0]) * iw
			boxes[:,3] = boxes[:,3] / float(input_shape[1]) * ih
			for b in boxes:
				thld_boxes.append(b)
		if len(thld_boxes) != 0: 
			# nms
			boxes=deal(torch.stack(thld_boxes))
			for b in boxes:
				cx = b[0]
				cy = b[1]
				w = b[2]
				h = b[3]
				draw.rectangle([cx-w/2, cy-h/2, cx+w/2, cy+h/2], outline='red')
				draw.text((cx-w/2, cy-h/2+11), str(b[4].long().item()), fill="#360066")
				f.write(str(b[4].long().item())+' '+str(b[5].item())+'\n')
		del draw
		raw.save('result/image/'+line[:-1].split('/')[-1])
