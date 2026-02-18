import torch
import torch.nn as nn
from utils import intersect_dicts
import HourglassPvt2_Base
import torch.nn.functional as F
import math
from skimage.segmentation import slic
import numpy as np

class LocalAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(LocalAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(dim, num_heads=4)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, self.window_size * self.window_size).permute(2, 0, 1)
        attn_output, _ = self.attention(x, x, x)  # [16, 2304, 384])
        attn_output = attn_output.permute(1, 2, 0).view(B, H // self.window_size, W // self.window_size, C, self.window_size, self.window_size) # [1, 48, 48, 384, 4, 4])
        attn_output = attn_output.permute(0, 3, 1, 4, 2, 5).contiguous() # ([1, 384, 48, 4, 48, 4])
        attn_output = attn_output.view(B, C, H, W) # [1, 384, 192, 192]
        return attn_output

class ImprovedModule(nn.Module):
    def __init__(self, dim):
        super(ImprovedModule, self).__init__()
        self.local_attn = LocalAttention(dim, window_size=4)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.local_attn(x)
        x = self.relu(self.conv(x))
        return x

class DGFU(nn.Module):
    """
    GCN Layer with dynamic SLIC-based superpixel indexing.
    Suitable for camouflage object detection (COD) or region-aware tasks.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 block_num,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 padding_mode='zeros',
                 adj_mask=None):
        super(DGFU, self).__init__()

        self.Conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                bias,
                                padding_mode)

        self.in_features = in_channels
        self.out_features = out_channels
        self.block_num = block_num
        self.adj_mask = adj_mask

        self.W = nn.Parameter(torch.randn(in_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)

    def generate_slic_index(self, feature_map, n_segments=100):
        """
        Generate superpixel index using SLIC for a single feature map.
        Input: feature_map (C, H, W)
        Output: index (1, H, W)
        """
        fmap_np = feature_map.detach().cpu().numpy()
        fmap_np = np.transpose(fmap_np, (1, 2, 0))  # (H, W, C)
        segments = slic(fmap_np, n_segments=n_segments, compactness=10, sigma=1, start_label=0)
        segments_tensor = torch.from_numpy(segments).unsqueeze(0).long()  # (1, H, W)
        return segments_tensor

    def forward(self, input):
        """
        input: Tensor of shape (B, C, H, W)
        """
        batch_size, channels, h, w = input.shape
        device = input.device

        # --- Step 1: Generate SLIC-based index ---
        index_list = []
        for i in range(batch_size):
            idx = self.generate_slic_index(input[i], n_segments=self.block_num)
            index_list.append(idx.to(device))
        index = torch.cat(index_list, dim=0).unsqueeze(1)  # (B, 1, H, W)

        # --- Step 2: Upsample index if needed ---
        index = F.interpolate(index.float(), size=(h, w), mode='nearest').long()

        # --- Step 3: Build one-hot index ---
        index_ex = torch.zeros(batch_size, self.block_num, h, w, device=device)
        index_ex = index_ex.scatter_(1, index, 1)
        block_value_sum = torch.sum(index_ex, dim=(2, 3))

        # --- Step 4: Regional mean feature (graph node features) ---
        input_ = input.repeat(self.block_num, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)  # (B, block_num, C, H, W)
        index_ex = index_ex.unsqueeze(2)
        input_means = torch.sum(index_ex * input_, dim=(3, 4)) / \
                      (block_value_sum + (block_value_sum == 0).float()).unsqueeze(2)

        # --- Step 5: Compute adjacency matrix ---
        input_means_ = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 2, 0, 3)
        input_means_ = (input_means_ - input_means.unsqueeze(1)).permute(0, 2, 1, 3)

        M = self.W @ self.W.T  # (C, C)
        adj = input_means_.reshape(batch_size, -1, channels).matmul(M)
        adj = torch.sum(adj * input_means_.reshape(batch_size, -1, channels), dim=2)
        adj = adj.view(batch_size, self.block_num, self.block_num)
        adj = torch.exp(-1 * adj)

        if self.adj_mask is not None:
            adj = adj * self.adj_mask.to(device)

        # --- Step 6: Graph aggregation ---
        adj_means = input_means.repeat(self.block_num, 1, 1, 1).permute(1, 0, 2, 3) * adj.unsqueeze(3)
        adj_means = (1 - torch.eye(self.block_num, device=device).reshape(1, self.block_num, self.block_num, 1)) * adj_means
        adj_means = torch.sum(adj_means, dim=2)

        # --- Step 7: Feature update ---
        features = torch.sum(index_ex * (input_ + adj_means.unsqueeze(3).unsqueeze(4)), dim=1)
        output = self.Conv2d(features)
        return output

    def __repr__(self):
        return self.__class__.__name__ + f' ({self.in_features} -> {self.out_features})'

class RGMM(nn.Module):
    def __init__(self, dim_in=32, dim_mid=16, mids=4, img_size=384, up=False):
        super(RGMM, self).__init__()

        self.normalize = nn.LayerNorm(dim_in)
        self.img_size = img_size
        self.num_s = int(dim_mid)
        self.num_n = (mids * mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(dim_in, self.num_s, kernel_size=1)
        self.gcn = DGFU(
            in_channels=self.num_s,
            out_channels=self.num_s,
            kernel_size=1,
            block_num=16,#8,10,16,18,20
            bias=False
        )
        self.conv_extend = nn.Conv2d(self.num_s, dim_in, kernel_size=1, bias=False)

        if up:
            self.local_attn = ImprovedModule(dim_in)
        else:
            self.local_attn = nn.Identity()

    def forward(self, x1, x2):
        n, c, h, w = x1.size()
        x2 = self.local_attn(x2)
        x2 = F.softmax(x2, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x1).view(n, self.num_s, -1)  # 1x1conv downsampling
        x_proj = self.conv_proj(x1)
        x_mask = x_proj * x2

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = F.softmax(x_proj_reshaped, dim=1)

        x_rproj_reshaped = x_proj_reshaped

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # Reshape to image-like format for Graph2dConvolution
        x_gcn_input = x_n_state.permute(0, 2, 1).unsqueeze(3)  # (B, C, N, 1)
        x_gcn_output = self.gcn(x_gcn_input)
        x_n_rel = x_gcn_output.squeeze(3).permute(0, 2, 1)  # (B, N, C)

        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x2.size()[2:])
        out = x1 + (self.conv_extend(x_state))

        return out

class UpSampling2x(nn.Module):
    def __init__(self, in_chs, out_chs):
        super(UpSampling2x, self).__init__()
        temp_chs = out_chs * 4  # for PixelShuffle
        self.up_module = nn.Sequential(
            nn.Conv2d(in_chs, temp_chs, 1, bias=False),
            nn.BatchNorm2d(temp_chs),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)  # (B, C*r*r, H，w) reshape to (B, C, H*r，w*r)
        )

    def forward(self, features):
        return self.up_module(features)

class DFF(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool2d(4)


        self.conv_atten1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_atten3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


        self.gate_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )


        self.conv_redu = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)


        self.residual_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)


        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(dim, 1, kernel_size=1, stride=1, bias=True)

        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):

        output = torch.cat([x, skip], dim=1)


        att1 = self.conv_atten1(self.avg_pool1(output))
        att2 = self.conv_atten2(self.avg_pool2(output))
        att3 = self.conv_atten3(self.avg_pool3(output))


        att2 = nn.functional.interpolate(att2, size=att1.size()[2:], mode='bilinear', align_corners=True)
        att3 = nn.functional.interpolate(att3, size=att1.size()[2:], mode='bilinear', align_corners=True)


        att = att1 + att2 + att3


        gate = self.gate_conv(output)
        att = att * gate


        output = output * att


        output = self.conv_redu(output)


        residual = self.residual_conv(x)
        output = output + residual


        att = self.conv1(x) + self.conv2(skip)
        att = self.nonlin(att)


        output = output * att
        return output
class ACFFM(nn.Module):
    def __init__(self, in_chs, out_chs, end=False):
        super(ACFFM, self).__init__()

        self.dff1 = DFF(in_chs)
        self.dff2 = DFF(in_chs)
        self.dff3 = DFF(in_chs)

        if end:
            tmp_chs = in_chs * 2
        else:
            tmp_chs = in_chs

        # Feature fusion block 1
        self.gf1 = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, 1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chs, in_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True)
        )

        # Feature fusion block 2
        self.gf2 = nn.Sequential(
            nn.Conv2d(in_chs, tmp_chs, 1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True)
        )

        # Feature fusion block 3
        self.gf3 = nn.Sequential(
            nn.Conv2d(in_chs, tmp_chs, 1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(tmp_chs, tmp_chs, 3, padding=1, bias=False),
            nn.BatchNorm2d(tmp_chs),
            nn.ReLU(inplace=True)
        )

        self.up2x_1 = UpSampling2x(tmp_chs, out_chs)
        self.up2x_2 = UpSampling2x(tmp_chs, out_chs)

    def forward(self, f_up1, f_up2, f_down1, f_down2):
        # fc1 = torch.cat((f_down1, f_down2), dim=1)
        # f_tmp = self.gf1(fc1)
        #
        # out1 = self.gf2(torch.cat((f_tmp, f_up1), dim=1))
        # out2 = self.gf3(torch.cat((f_tmp, f_up2), dim=1))
        #
        # return self.up2x_1(out1), self.up2x_2(out2)
        # 使用 DFF 替代 torch.cat
        f_tmp = self.dff1(f_down1, f_down2)  # 融合下采样特征
        f_tmp = self.gf1(f_tmp)

        out1 = self.gf2(self.dff2(f_tmp, f_up1))  # 融合中间特征
        out2 = self.gf3(self.dff3(f_tmp, f_up2))

        return self.up2x_1(out1), self.up2x_2(out2)

class OutPut(nn.Module):
    def __init__(self, in_chs):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(UpSampling2x(in_chs, in_chs),
                                 nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid()
                                 )

    def forward(self, feat):
        return self.out(feat)

class Net(nn.Module):
    def __init__(self, ckpt_path, img_size=384):
        super(Net, self).__init__()
        self.encoder = HourglassPvt2_Base.Hourglass_vision_transformer_base_v2()
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            csd = intersect_dicts(ckpt, self.encoder.state_dict())  # intersect
            msg = self.encoder.load_state_dict(csd, strict=False)  # load

            print("====================================")
            pt_name = ckpt_path.split('/')[-1]
            print(f'Transferred {len(csd)}/{len(self.encoder.state_dict())} items from {pt_name}')

        self.gf1_1 = ACFFM(320, 128)
        self.gf1_2 = ACFFM(128, 64)
        self.gf1_3 = ACFFM(64, 64, end=True)

        self.gf2_2 = ACFFM(128, 64)
        self.gf2_3 = ACFFM(64, 64, end=True)

        self.out_F1 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )

        self.out_F2 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True)
                                    )

        self.sam1 = RGMM(128, 16, up=False)
        self.sam2 = RGMM(128, 16, up=False)
        self.conv_f = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1,  bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True)
                                    )

        self.out1 = OutPut(in_chs=128)
        self.out2 = OutPut(in_chs=128)
        self.out3 = OutPut(in_chs=256)

    def cim_decoder(self, tokens):
        f = []
        size = [96, 96, 48, 48, 24, 24, 24, 24, 48, 48, 96, 96]
        for i in range(len(tokens)):  # [b,576,768] [b,2304,384]
            b, _, c = tokens[i].shape
            f.append(tokens[i].permute(0, 2, 1).view(b, c, size[i], size[i]).contiguous())

        f1_1, f1_2 = self.gf1_1(f[7], f[4], f[5], f[6])

        f2_1, f2_2 = self.gf1_2(f[9], f[8], f1_1, f1_2)
        f2_3, f2_4 = self.gf2_2(f[3], f[2], f1_1, f1_2)

        f3_1, f3_2 = self.gf1_3(f[11], f[10], f2_2, f2_1)
        f3_3, f3_4 = self.gf2_3(f[1], f[0], f2_3, f2_4)

        fout1 = self.out_F1(torch.cat([f3_1, f3_2], dim=1))
        fout2 = self.out_F2(torch.cat([f3_3, f3_4], dim=1))

        return fout1, fout2  # high, low

    def pred_outs(self, gpd_outs):
        return [self.out1(gpd_outs[0]), self.out2(gpd_outs[1]), self.out3(gpd_outs[2])]

    def forward(self, img):
        #
        B, C, H, W = img.size()
        x = self.encoder(img)  # include

        out_high, out_low = self.cim_decoder(x)
        out1 = self.sam1(out_high, out_low)
        out2 = self.sam2(out_low, out_high)
        out_f = self.conv_f(torch.cat([out1, out2], dim=1))

        out = self.pred_outs([out2, out1, out_f])  # low, high, fusion

        return out





if __name__=='__main__':
    from thop import profile
    x = torch.rand(1, 3, 384, 384)
    net = Net(None)
    flops, params = profile(net, (x,))
    print(f"Flops: {flops / 1e9:.4f} GFlops")
    print(f"Params: {params / 1e6:.4f} MParams")

