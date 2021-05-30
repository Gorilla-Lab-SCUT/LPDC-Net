import torch
import torch.nn as nn
from im2mesh.layers import ResnetBlockFC


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maxpooling operation.

    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=512):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))
        return c


class SpatioTemporalResnetPointnet(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=512,
                 use_only_first_pcl=False, pool_once=False, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.use_only_first_pcl = use_only_first_pcl
        self.pool_once = pool_once

        self.spatial_fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.spatial_block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_fc_c = nn.Linear(hidden_dim, c_dim)

        if pool_once:
            self.temporal_block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
            self.temporal_block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
            self.temporal_block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
            self.temporal_block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
            self.temporal_block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        else:
            self.temporal_fc_pos = nn.Linear(dim+1, 3*hidden_dim)
            self.temporal_block_0 = ResnetBlockFC(3*hidden_dim, hidden_dim)
            self.temporal_block_1 = ResnetBlockFC(3*hidden_dim, hidden_dim)
            self.temporal_block_2 = ResnetBlockFC(3*hidden_dim, hidden_dim)
            self.temporal_block_3 = ResnetBlockFC(3*hidden_dim, hidden_dim)
            self.temporal_block_4 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_fc_c = nn.Linear(hidden_dim, c_dim)
        self.fc_c = nn.Linear(2 * c_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        batch_size, n_steps, n_pts, n_dim = x.shape
        t = (torch.arange(
            n_steps, dtype=torch.float32) / (n_steps - 1)
        ).to(x.device)
        t = t[None, :, None, None].expand(batch_size, n_steps, n_pts, 1)
        x_t = torch.cat([x, t], dim=3).reshape(batch_size, n_steps, n_pts, n_dim+1)

        net = self.spatial_fc_pos(x)
        net = self.spatial_block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_4(net)
        net = self.pool(net, dim=2)
        spatial_c = self.spatial_fc_c(self.actvn(net)) #batch_size x n_steps x c_dim

        net = self.temporal_fc_pos(x_t)
        net = self.temporal_block_0(net)
        if self.pool_once:
            pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=3)
        else:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled_time = self.pool(pooled, dim=1, keepdim=True)
            net = torch.cat([net, pooled.expand(net.size()), pooled_time.expand(net.size())], dim=3)

        net = self.temporal_block_1(net)
        if self.pool_once:
            pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=3)
        else:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled_time = self.pool(pooled, dim=1, keepdim=True)
            net = torch.cat([net, pooled.expand(net.size()), pooled_time.expand(net.size())], dim=3)

        net = self.temporal_block_2(net)
        if self.pool_once:
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2)
        else:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled_time = self.pool(pooled, dim=1, keepdim=True)
            net = torch.cat([net, pooled.expand(net.size()), pooled_time.expand(net.size())], dim=3)

        net = self.temporal_block_3(net)
        if self.pool_once:
            pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=2)
        else:
            pooled = self.pool(net, dim=2, keepdim=True)
            pooled_time = self.pool(pooled, dim=1, keepdim=True)
            net = torch.cat([net, pooled.expand(net.size()), pooled_time.expand(net.size())], dim=3)

        net = self.temporal_block_4(net)
        net = self.pool(net, dim=2)  
        temporal_c = self.temporal_fc_c(self.actvn(net)) #batch_size x n_steps x c_dim

        spatiotemporal_c = torch.cat([spatial_c, temporal_c], dim=2)
        spatiotemporal_c = self.fc_c(spatiotemporal_c) #batch_size x n_steps x c_dim
        return spatial_c, spatiotemporal_c

class SpatioTemporalResnetPointnet2(nn.Module):

    def __init__(self, c_dim=128, dim=3, hidden_dim=512,
                 use_only_first_pcl=False, **kwargs):
        super().__init__()
        self.c_dim = c_dim
        self.use_only_first_pcl = use_only_first_pcl

        self.spatial_fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.spatial_block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.spatial_fc_c = nn.Linear(hidden_dim, c_dim)

        self.temporal_fc_pos = nn.Linear(dim+1, 3*hidden_dim)
        self.temporal_block_0 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_block_1 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_block_2 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_block_3 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_block_4 = ResnetBlockFC(3*hidden_dim, hidden_dim)
        self.temporal_fc_c = nn.Linear(2* hidden_dim, c_dim)

        self.fc_c = nn.Linear(2 * c_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, x):
        batch_size, n_steps, n_pts, n_dim =  x.shape
        t = (torch.arange(
            n_steps, dtype=torch.float32) / (n_steps - 1)
        ).to(x.device)
        t = t[None, :, None, None].expand(batch_size, n_steps, n_pts, 1)
        x_t = torch.cat([x, t], dim=3).reshape(batch_size, n_steps, n_pts, n_dim+1)

        net = self.spatial_fc_pos(x)
        net = self.spatial_block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=3)

        net = self.spatial_block_4(net)
        net = self.pool(net, dim=2)
        spatial_c = self.spatial_fc_c(self.actvn(net)) #batch_size x n_steps x c_dim

        net = self.temporal_fc_pos(x_t)
        net = self.temporal_block_0(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_1(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_2(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_3(net)
        pooled = self.pool(net, dim=2, keepdim=True)
        pooled_expand = pooled.expand(net.size())
        pooled2 = self.pool(pooled, dim=1, keepdim=True)
        pooled2_expand = pooled2.expand(net.size())
        net = torch.cat([net, pooled_expand, pooled2_expand], dim=3)

        net = self.temporal_block_4(net)
        net = self.pool(net, dim=2)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)
        temporal_c = self.temporal_fc_c(self.actvn(net)) #batch_size x n_steps x c_dim

        spatiotemporal_c = torch.cat([spatial_c, temporal_c], dim=2)
        spatiotemporal_c = self.fc_c(self.actvn(spatiotemporal_c)) #batch_size x n_steps x c_dim
        return spatiotemporal_c, spatiotemporal_c