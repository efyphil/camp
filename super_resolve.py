from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import ToTensor

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
#img = Image.open(opt.input_image).convert('YCbCr')
img = Image.open(opt.input_image)
y, cb, cr = img.split()

model = torch.load(opt.model)
img_to_tensor = ToTensor()
input = img_to_tensor(img).view(-1, 3, img.size[1], img.size[0])

if opt.cuda:
    model = model.cuda()
    input = input.cuda()

out = model(input)
out = out.cpu()

y = out[0][0].detach().numpy()
cb = out[0][1].detach().numpy()
cr = out[0][2].detach().numpy()

# out_img_y = y.detach().numpy()
out_img_y = y
out_img_y *= 255.0
out_img_y = out_img_y.clip(0, 255)
out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

# out_img_cb = cb.detach().numpy()
out_img_cb = cb
out_img_cb *= 255.0
out_img_cb = out_img_cb.clip(0, 255)
out_img_cb = Image.fromarray(np.uint8(out_img_cb), mode='L')

# out_img_cr = cr.detach().numpy()
out_img_cr = cr
out_img_cr *= 255.0
out_img_cr = out_img_cr.clip(0, 255)
out_img_cr = Image.fromarray(np.uint8(out_img_cr), mode='L')

#out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
#out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)

out_img = Image.merge('RGB', [out_img_y, out_img_cb, out_img_cr])
#out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

out_img.save(opt.output_filename)
print('output image saved to ', opt.output_filename)
