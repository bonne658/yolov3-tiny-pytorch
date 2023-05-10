import torch
import torch.nn as nn
from base import BaseConv
import numpy as np

class Backbone(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv0 = BaseConv(3, 16, 3, 1, 1, True, 'leaky')
		self.maxpool1 = nn.MaxPool2d(2)
		self.conv2 = BaseConv(16, 32, 3, 1, 1, True, 'leaky')
		self.maxpool3 = nn.MaxPool2d(2)
		self.conv4 = BaseConv(32, 64, 3, 1, 1, True, 'leaky') 
		self.maxpool5 = nn.MaxPool2d(2)
		self.conv6 = BaseConv(64, 128, 3, 1, 1, True, 'leaky')
		self.maxpool7 = nn.MaxPool2d(2)
		self.conv8 = BaseConv(128, 256, 3, 1, 1, True, 'leaky')
		self.maxpool9 = nn.MaxPool2d(2)
		self.conv10 = BaseConv(256, 512, 3, 1, 1, True, 'leaky')
		self.pad = nn.ZeroPad2d((0,1,0,1))
		self.maxpool11 = nn.MaxPool2d(2, 1)
		self.conv12 = BaseConv(512, 1024, 3, 1, 1, True, 'leaky')
		self.conv13 = BaseConv(1024, 256, 1, 1, 0, True, 'leaky')
		self.conv14 = BaseConv(256, 512, 3, 1, 1, True, 'leaky')
	def forward(self, x):
		x0=self.conv0(x)
		x1=self.maxpool1(x0)  # 1    16*208*208
		x2=self.conv2(x1)
		x3=self.maxpool3(x2)  # 3    32*104*104
		x4=self.conv4(x3)
		x5=self.maxpool5(x4)  # 5    64*52*52
		x6=self.conv6(x5)
		x7=self.maxpool7(x6)  # 7    128*26*26
		r2=self.conv8(x7)
		x9=self.maxpool9(r2)  # 9    256*13*13
		x10=self.conv10(x9)   # 10   512*13*13
		x11=self.pad(x10)
		x11=self.maxpool11(x11)       # 11     512*13*13
		x12=self.conv12(x11)          # 12     1024*13*13
		r1=self.conv13(x12)           # 13     256*13*13
		x14=self.conv14(r1)           # 14     512*13*13
		return r1, r2, x14		

class Tiny(nn.Module):
	def __init__(self, class_num):
		super().__init__()
		self.backbone = Backbone()
		cc = 3*(5+class_num)
		self.convs=[]
		for m in self.backbone.modules():
			if isinstance(m, BaseConv):
				self.convs.append(m)
		self.conv15 = BaseConv(512, cc, 1, 1, 0, False, 'linear')
		self.conv18 = BaseConv(256, 128, 1, 1, 0, True, 'leaky')
		self.upsample19 = nn.Upsample(scale_factor=2)
		self.conv21 = BaseConv(384, 256, 3, 1, 1, True, 'leaky')
		self.conv22 = BaseConv(256, cc, 1, 1, 0, False, 'linear')
		self.convs.append(self.conv15)
		self.convs.append(self.conv18)
		self.convs.append(self.conv21)
		self.convs.append(self.conv22)
		
	def forward(self, x):
		r1, r2, x14 = self.backbone(x)
		out1=self.conv15(x14)         # 15     255*13*13
		x18=self.conv18(r1)           # 18     128*13*13
		x19=self.upsample19(x18)      # 19     128*26*26
		x20=torch.cat([x19, r2], 1)   # 20     384*26*26
		x21=self.conv21(x20)          # 21     256*26*26
		out2=self.conv22(x21)         # 22     255*26*26
		return out1, out2
		
	def load_darknet(self, weights_path):
		with open(weights_path, "rb") as f:
			header = np.fromfile(f, dtype=np.int32, count=5)
			weights = np.fromfile(f, dtype=np.float32)
		ptr = 0
		for i,conv in enumerate(self.convs):
			conv_layer = conv.conv
			if i!=9 and i!=12:
				bn_layer = conv.bn
				num_b = bn_layer.bias.numel()
				# bn bias
				bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
				bn_layer.bias.data.copy_(bn_b)
				ptr += num_b
				# bn weight
				bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
				bn_layer.weight.data.copy_(bn_w)
				ptr += num_b
				# Running Mean
				bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
				bn_layer.running_mean.data.copy_(bn_rm)
				ptr += num_b
				# Running Var
				bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
				bn_layer.running_var.data.copy_(bn_rv)
				ptr += num_b
			else:
				# conv bias
				num_b = conv_layer.bias.numel()
				conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
				conv_layer.bias.data.copy_(conv_b)
				ptr += num_b
			# conv
			num_w = conv_layer.weight.numel()
			conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
			conv_layer.weight.data.copy_(conv_w)
			ptr += num_w
			rest = weights.size - ptr
			if rest <= 0 : 
				print(i)
				break
		print('rest: ', rest)
		return rest
		
	def init(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight.data, 0.0, 0.02)
			if isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight.data, 1.0, 0.02)
				nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
	data=torch.zeros(2,3,416,416)
	t=Tiny()
	x1, x2=t(data)
	print(x1.shape, x2.shape)
	print(t.load_darknet('/home/lwd/code/darknet/yolov3-tiny.weights'))
