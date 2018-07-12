import torch
import torch.nn as nn
import torch.nn.init as init



class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(3, 64, (5, 5), stride=(1, 1), padding=(2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv21 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv22 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2 * 3, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle1 = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle1(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)

# class Conv_ReLU_Block(nn.Module):
#     def __init__(self):
#         super(Conv_ReLU_Block, self).__init__()
#         self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         return self.relu(self.conv(x))
        
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
#         self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.relu = nn.ReLU(inplace=True)
    
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, sqrt(2. / n))
                
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         residual = x
#         out = self.relu(self.input(x))
#         out = self.residual_layer(out)
#         out = self.output(out)
#         out = torch.add(out,residual)
#         return out