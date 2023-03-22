# Author: Zhaoxin Fan, Zhenbo Song

import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from models.resnet import ResNetBase
import torch


class csvt_module(nn.Module):
    def __init__(self, channels, num_tokens=16):
        super(csvt_module, self).__init__()

        # layers for generate tokens
        self.q_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(channels, num_tokens, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # layers for tranformer
        self.convvalues = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convkeys = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convquries = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.embedding1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # layers for projector
        self.p_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.T_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # hidden state
        self.trans_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # generate tokens
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        
        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]    # N*C
            dk = x_k.F[start_id:end_id, :].T  # num_tokens*N
            dk = self.softmax(dk)             # N*num_tokens
         
            de = torch.matmul(dk, dq).T       # C*num_tokens
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            de = torch.unsqueeze(de, dim=0)
            x_feat.append(de)
            start_id = end_id
        tokens = torch.cat(x_feat, dim=0)     # B*C*num_tokens

        # visul transormers on multi tokens
        vt_values = self.convvalues(tokens)  
        vt_keys = self.convkeys(tokens)                            # B*C*num_tokens
        vt_quires = self.convquries(tokens)                        # B*C*num_tokens
        vt_map = torch.matmul(vt_keys.transpose(1, 2), vt_quires)  # B*num_tokens*num_tokens
        vt_map = self.softmax(vt_map)                              # B*num_tokens*num_tokens
        T_middle = torch.matmul(vt_map, vt_values.transpose(1, 2)).transpose(1, 2)  # B*C*num_tokens
        #T_out = tokens + self.actembedding1(self.bnembedding1(self.embedding1(T_middle)))                    # B*C*num_tokens
        T_out = tokens + self.embedding1(T_middle) 

        # projector
        x_p = self.p_conv(x)
        T_P = self.T_conv(T_out)

        start_id = 0
        x_feat2 = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dp = x_p.F[start_id:end_id, :]   # N*C
            dt = T_P[i]                        # C*num_tokens
 
            dm = torch.matmul(dp, dt)        # N*num_tokens
            dm = self.softmax(dm)            # N*num_tokens
           
            df = torch.matmul(dm, dt.T)      # N*C
            x_feat2.append(df)
            start_id = end_id
        x_r = torch.cat(x_feat2, dim=0)

        x_r = ME.SparseTensor(coordinates=x.coordinates, features=x_r,
                                    coordinate_map_key=x.coordinate_map_key,
                                    coordinate_manager=x.coordinate_manager)
        x_r = x+ self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r


class asvt_module(nn.Module):
    def __init__(self, channels, reduction=8):
        super(asvt_module, self).__init__()
        self.q_conv = ME.MinkowskiConvolution(channels, channels // reduction, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(channels, channels // reduction, 1, dimension=3, bias=False)

        self.v_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.trans_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]    # N*C
            dk = x_k.F[start_id:end_id, :].T  # C*N
            dv = x_v.F[start_id:end_id, :]    # N*C
            de = torch.matmul(dq, dk)         # N*N
            da = self.softmax(de)             # N*N
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            dr = torch.matmul(da, dv)         # N*C
            x_feat.append(dr)
            start_id = end_id
        x_r = torch.cat(x_feat, dim=0)
        x_r = ME.SparseTensor(coordinates=x.coordinates, features=x_r,
                              coordinate_map_key=x.coordinate_map_key,
                              coordinate_manager=x.coordinate_manager)
        
        x_r = x+self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r


class SVTNet(ResNetBase):
    def __init__(self, in_channels, out_channels, num_top_down=1, conv0_kernel_size=5, block=BasicBlock,
                 layers=(1, 1, 1), planes=(32, 64, 64)):
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        assert 0 <= num_top_down <= len(layers)
        self.num_bottom_up = len(layers)
        self.num_top_down = num_top_down
        self.conv0_kernel_size = conv0_kernel_size
        self.block = block
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        ResNetBase.__init__(self, in_channels, out_channels, D=3)

    def network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()    # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()       # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()   # Bottom-up blocks
        self.tconvs = nn.ModuleList()   # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(in_channels, self.inplanes, kernel_size=self.conv0_kernel_size,
                                             dimension=D)
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D))
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        self.conv1x1.append(ME.MinkowskiConvolution(self.inplanes,self.lateral_dim, kernel_size=1,
                                                    stride=1, dimension=D))

        #before_lateral_dim=plane
        after_reduction = max(self.lateral_dim/ 8, 8)
        reduction = int(self.lateral_dim// after_reduction)
        
        self.asvt = asvt_module(self.lateral_dim ,reduction)
        self.csvt = csvt_module(self.lateral_dim, 8)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)     # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)
        x = self.conv1x1[0](x)
        x_csvt= self.csvt(x)

        x_asvt= self.asvt(x)
        x = x_csvt+x_asvt
        
        return x