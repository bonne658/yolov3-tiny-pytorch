import torch
import torch.nn as nn

class BaseConv(nn.Module):
	def __init__(self, in_channel, out_channel, k_size, stride, pad, bn, act):
		super().__init__()
		self.conv = nn.Conv2d(in_channel, out_channel, k_size, stride, pad, bias=not bn)
		if bn == True:
			self.bn = nn.BatchNorm2d(out_channel)
		else:
			self.bn = nn.Identity()
		if act == 'leaky': self.act = nn.LeakyReLU(0.1)
		elif act == 'linear': self.act = nn.Identity()
	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return self.act(x)
		
if __name__ == '__main__':
	data=torch.zeros(2,3, 320,320)
	bc=BaseConv(3, 16, 3, 1, 1, False, 'leaky')
	x = bc(data)
	print(x.shape)
