# Author: Jacek Komorowski
# Warsaw University of Technology

import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.resnet import ResNetBase


class MinkFPN(ResNetBase):
    # Feature Pyramid Network (FPN) architecture implementation using Minkowski ResNet building blocks
    # in_channels=1, out_channels=256, num_top_down = 1, conv0_kernel_size=5, layers=[1,1,1], planes=[32,64,64]
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock, layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)  # 3
        self.num_top_down = num_top_down  # 1
        self.conv0_kernel_size = conv0_kernel_size  # 5
        self.block = block  # BasicBlock
        self.layers = layers  # [1,1,1]
        self.planes = planes  # [32,64,64]
        self.lateral_dim = out_channels  # 256
        self.init_dim = planes[0]  # 32
        # in_channels=1, out_channels=256
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]  # 32
        # in_channels=1, out_channels=32, kernel_size=5, dimension=3
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)  # self.inplanes=32

        for plane, layer in zip(self.planes, self.layers):
            # self.inplanes = 32
            self.convs.append(ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            # self.inplanes = 32
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            # self.block=BasicBlock， plane=[32,64,64], layer=[1,1,1]
            self.blocks.append(self._make_layer(self.block, plane, layer))

        # Lateral connections
        # num_top_down=1
        for i in range(self.num_top_down):
            # in=self.planes[-1 - 0]=64, out=256
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - i], self.lateral_dim, kernel_size=1, stride=1, dimension=D))
            # in=256, out=256
            self.tconvs.append(ME.MinkowskiConvolutionTranspose(self.lateral_dim, self.lateral_dim, kernel_size=2, stride=2, dimension=D))
        # There's one more lateral connection than top-down TConv blocks
        # self.num_top_down=1, self.num_bottom_up=3,
        if self.num_top_down < self.num_bottom_up:
            # Lateral connection from Conv block 1 or above
            # # in=self.planes[-1 - 1]=64, out=256
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[-1 - self.num_top_down], self.lateral_dim, kernel_size=1, stride=1, dimension=D))
        else:
            # Lateral connection from Con0 block
            self.conv1x1.append(ME.MinkowskiConvolution(self.planes[0], self.lateral_dim, kernel_size=1, stride=1, dimension=D))

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):  # coor=([49778, 4]), feat=([49778, 1])
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        feature_maps = []
        x = self.conv0(x)  # coor=([49778, 4]), feat=([49778, 32])
        x = self.bn0(x)
        x = self.relu(x)
        if self.num_top_down == self.num_bottom_up:
            feature_maps.append(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Decreases spatial resolution (conv stride=2) torch.Size([31895, 32]), torch.Size([11530, 32]), torch.Size([3951, 64])
            x = bn(x)
            x = self.relu(x)
            x = block(x)        # ([31895, 32])
            # self.num_top_down=1, self.num_bottom_up=3, (3-1-1)=1 <= ndx < (3-1)=2
            if self.num_bottom_up - 1 - self.num_top_down <= ndx < len(self.convs) - 1:
                feature_maps.append(x)

        assert len(feature_maps) == self.num_top_down
        # x:([3951, 64])
        x = self.conv1x1[0](x)  # ([3951, 256])

        # TOP-DOWN PASS
        for ndx, tconv in enumerate(self.tconvs):
            x = tconv(x)  # Upsample using transposed convolution   # ([11530, 256])
            x = x + self.conv1x1[ndx + 1](feature_maps[-ndx - 1])   # [11530, 256])

        return x
