import torch, math, sys
import numpy as np
import torch.nn as nn

def iou(a,b):
	A=len(a)
	B=len(b)
	area1=a[:,2]*a[:,3]
	area1=area1.unsqueeze(1).expand(A,B)
	area2=b[:,2]*b[:,3]
	area2=area2.unsqueeze(0).expand(A,B)
	aa=torch.zeros_like(a)
	aa[:,0:2]=a[:,0:2]-a[:,2:4]/2
	aa[:,2:4]=aa[:,0:2]+a[:,2:4]
	aa=aa.unsqueeze(1).expand(A,B,4)
	bb=torch.zeros_like(b)
	bb[:,0:2]=b[:,0:2]-b[:,2:4]/2
	bb[:,2:4]=bb[:,0:2]+b[:,2:4]
	bb=bb.unsqueeze(0).expand(A,B,4)
	lt=torch.max(aa[:,:,0:2], bb[:,:,0:2])
	rb=torch.min(aa[:,:,2:4], bb[:,:,2:4])
	inter=torch.clamp((rb-lt), min=0)
	inter=inter[:,:,0]*inter[:,:,1]
	return inter/(area1+area2-inter)

def clip(a):
	mi=1e-7
	ma=1-mi
	b=(a>=mi).float()*a+(a<mi).float()*mi
	b=(b<=ma).float()*b+(b>ma).float()*ma
	return b
	
def BCELoss(pred, target):
	p=clip(pred)
	return -target*torch.log(p)-(1-target)*torch.log(1-p)
	
def MSELoss(pred, target):
	return torch.pow((pred-target), 2)

class Loss(nn.Module):
	def __init__(self, input_size, anchors, classes, anchors_mask=[[0,1,2], [3,4,5]]):
		super().__init__()
		self.input_size = input_size
		self.anchors = anchors
		self.bbox_attrs = 5 + classes
		self.anchors_mask = anchors_mask
		self.ignore_threshold = 0.05
		
	'''
	l:            第l组anchors_mask
	out：         b*255*h*w， 网络输出之一
	targets:      b*N*5，比例形式的gt
	'''
	def forward(self, l, out, target):
		b = out.size(0)
		in_h = out.size(2)
		in_w = out.size(3)
		s = self.input_size[0] // in_w
		scaled_anchors = [(aw/s, ah/s) for aw,ah in self.anchors]
		# 正样本
		y_true, no_obj, scale = self.get_target(l, target, scaled_anchors, in_h, in_w)
		scale=2-scale
		out = out.view(b, 3, self.bbox_attrs, in_h, in_w).permute(0,1,3,4,2)
		x = torch.sigmoid(out[...,0])
		y = torch.sigmoid(out[...,1])
		w = out[...,2]
		h = out[...,3]
		# 记得sigmoid
		c = torch.sigmoid(out[...,4])
		cl=torch.sigmoid(out[...,5:])
		# 负样本
		no_obj = self.get_ignore(l,x,y,h,w,target, scaled_anchors, in_h, in_w, no_obj)
		if x.is_cuda:
			y_true = y_true.cuda()
			no_obj = no_obj.cuda()
			scale = scale.cuda()
		# loss
		xloss=torch.sum(BCELoss(x, y_true[...,0])*y_true[...,4]*scale)
		yloss=torch.sum(BCELoss(y, y_true[...,1])*y_true[...,4]*scale)
		wloss=torch.sum(MSELoss(w, y_true[...,2])*y_true[...,4]*scale*0.5)
		hloss=torch.sum(MSELoss(h, y_true[...,3])*y_true[...,4]*scale*0.5)
		closs=torch.sum(BCELoss(c, y_true[...,4])*y_true[...,4] + BCELoss(c, y_true[...,4])*no_obj)
		clsloss=torch.sum(BCELoss(cl[y_true[...,4]==1], y_true[...,5:][y_true[...,4]==1]))
		loss = xloss + yloss + wloss + hloss + closs + clsloss
		num=torch.sum(y_true[...,4])
		num=torch.max(num, torch.ones_like(num))
		# print(torch.sum(y_true[0,...,4]).item())
		# print(torch.sum(y_true[1,...,4]).item())
		#sys.exit()
		return loss, num
		
	'''
	l:            第l组anchors_mask
	targets:      b*N*5，比例形式的gt
	anchors:      9*2，已经放缩过的
	in_h：        特征图高度
	in_w：        特征图宽度
	每个batch：
		N*4的gt和9*4的anchor求iou
		每个gt的最大IOU对应的anchor：
			如果不在当前mask： continue
			否则：gt中心点坐标和anchor序号确定位置，赋值
	'''
	def get_target(self, l, targets, anchors, in_h, in_w):
		b = len(targets)
		c = len(self.anchors_mask[l])
		y_true = torch.zeros(b,c,in_h, in_w,self.bbox_attrs,requires_grad = False)
		no_obj = torch.ones(b,c,in_h, in_w,requires_grad = False)
		scale = torch.zeros(b,c,in_h, in_w,requires_grad = False)
		# 
		for bi in range(b):
			if(len(targets[bi]) == 0): continue
			# gt和anchors以(0,0)为中心计算iou
			batch_target = torch.zeros(len(targets[bi]), 4)
			batch_target[:,2] = targets[bi][:,2] * in_w
			batch_target[:,3] = targets[bi][:,3] * in_h
			anchor4 = torch.zeros(len(anchors), 4)
			anchor4[:,2:] = torch.FloatTensor(anchors)
			ious = iou(batch_target, anchor4)  # N * 9
			bests = torch.argmax(ious, dim=1)  # 每个值在0~8之间
			#print(bests)
			# 1.忘记赋值
			batch_target[:,0] = targets[bi][:,0] * in_w
			batch_target[:,1] = targets[bi][:,1] * in_h
			for it, best in enumerate(bests):
				if best not in self.anchors_mask[l]:
					continue
				c = self.anchors_mask[l].index(best)  # 0~2之间
				# gt中心点所在网格
				i = torch.floor(batch_target[it,0]).long()
				j = torch.floor(batch_target[it,1]).long()
				#print(bi,c,j,i)
				# 赋值
				no_obj[bi,c,j,i] = 0
				y_true[bi,c,j,i,0] = batch_target[it,0] - i.float()
				y_true[bi,c,j,i,1] = batch_target[it,1] - j.float()
				# 2.用错anchors(没放缩的self.anchors)
				y_true[bi,c,j,i,2] = math.log(batch_target[it,2]/anchors[best][0])
				y_true[bi,c,j,i,3] = math.log(batch_target[it,3]/anchors[best][1])
				y_true[bi,c,j,i,4] = 1
				clss=targets[bi][it][4].long()
				y_true[bi,c,j,i,5+clss] = 1
				scale[bi,c,j,i] = batch_target[it,2]*batch_target[it,3]/in_h/in_w
		return y_true, no_obj, scale
		
	'''
	l:            第l组anchors_mask
	x, y, h, w:   b*3*h*w，网络输出，其中x,y已经过sigmoid
	targets:      b*N*5，比例形式的gt
	anchors:      9*2，已经放缩过的
	in_h：        特征图高度
	in_w：        特征图宽度
	no_obj：      b*3*h*w，标记负样本
	将anchors_mask对应的anchors分布到特征图每个网格上，形状是b*3*h*w*2
	将x, y, h, w结合上面的anchors转化并concat成b*3*h*w*4的预测值
	每个batch：
		计算与gt的iou
		取每个预测框的最大iou值
		最大IOU超过阈值的是忽略样本，即no_obj对应的值设为0
	'''
	def get_ignore(self, l, x, y, h, w, targets, anchors, in_h, in_w, no_obj):
		ft = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		b = len(targets)
		# 转换h，w
		anchor_l = np.array(anchors)[self.anchors_mask[l]]
		anchor_w = ft(anchor_l[:,0:1])
		anchor_h = ft(anchor_l[:,1:])
		anchor_w = anchor_w.repeat(1,in_h*in_w).repeat(b,1).view(b,3,in_h,in_w)
		anchor_h = anchor_h.repeat(1,in_h*in_w).repeat(b,1).view(b,3,in_h,in_w)
		tw = (torch.exp(w.data)*anchor_w).unsqueeze(-1)
		th = (torch.exp(h.data)*anchor_h).unsqueeze(-1)
		# 转换x，y
		grid_y, grid_x = torch.meshgrid(torch.arange(in_w), torch.arange(in_h))
		# tensor可以这样转设备
		grid_x = grid_x.repeat(b,3,1,1).type(ft)
		grid_y = grid_y.repeat(b,3,1,1).type(ft)
		tx = (x.data + grid_x).unsqueeze(-1)
		ty = (y.data + grid_y).unsqueeze(-1)
		# concat
		pred = torch.cat([tx, ty, tw, th], -1)
		for bi in range(b):
			if(len(targets[bi]) == 0): continue
			# 计算iou
			pre = pred[bi].view(-1,4)
			# 形状，设备信息也一样
			gt = torch.zeros_like(targets[bi])
			gt[:,[0,2]] = targets[bi][:,[0,2]] * in_w
			gt[:,[1,3]] = targets[bi][:,[1,3]] * in_h
			gt = gt[:,:4]
			ious=iou(gt, pre)
			# 判断，赋值
			maxx, _ = torch.max(ious, dim=0)
			maxx = maxx.view(3,in_h,in_w)
			no_obj[bi][maxx > self.ignore_threshold] = 0
		return no_obj
