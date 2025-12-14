# DAU2Net.py
# (DA-U)²Net: Double-Attention U²-Net for Retinal Vessel Segmentation
# Implements: attention-gated skip connections, CBAM (SAM/CAM), dilated 4F blocks, deep supervision.
# Author: THESIS / GPT-5 Pro

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Small building utilities
# -------------------------

def conv3x3(in_ch, out_ch, dilation=1):
    padding = dilation
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Up2x(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, x, size=None, ref=None):
        if ref is not None:
            size = ref.shape[-2:]
        return F.interpolate(x, size=size, mode=self.mode, align_corners=False)


# -------------------------
# Attention modules
# -------------------------

class AttentionGate(nn.Module):
    """
    Feature-channel attention gate used in skip connections (paper Fig.2 / Fig.6).
    Given gating signal g and skip x, compute alpha in [0,1] and reweight x.
    """
    def __init__(self, in_x, in_g, inter_ch):
        super().__init__()
        self.theta_x = nn.Conv2d(in_x, inter_ch, kernel_size=1, bias=False)
        self.phi_g   = nn.Conv2d(in_g, inter_ch, kernel_size=1, bias=False)
        self.relu    = nn.ReLU(inplace=True)
        self.psi     = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        # match spatial sizes
        if x.shape[-2:] != g.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode='bilinear', align_corners=False)
        att = self.relu(self.theta_x(x) + self.phi_g(g))
        att = self.sigmoid(self.psi(att))
        return x * att


class ChannelAttention(nn.Module):
    """ CBAM-CAM (avg+max pool -> MLP -> sigmoid). """
    def __init__(self, in_ch, ratio=16):
        super().__init__()
        hidden = max(in_ch // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=(2,3), keepdim=True)
        mx, _ = torch.max(x, dim=2, keepdim=True)
        mx, _ = torch.max(mx, dim=3, keepdim=True)
        out = self.mlp(avg) + self.mlp(mx)
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """ CBAM-SAM (avg+max along channels -> 7x7 conv -> sigmoid). """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg, mx], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att


# -------------------------
# DARSU block (nested U)
# -------------------------

class DARSUBlock(nn.Module):
    """
    Double-Attention RSU-like block.
    - depth: number of encoder levels (e.g., 7,6,5,4). For '4F' we use dilated convs instead of pooling at the bottom.
    - high_res: if True, add SAM at the end (spatial attention) as in high-resolution modules; otherwise add CAM at end.
    """
    def __init__(self, in_ch, mid_ch, out_ch, depth=4, is_4f=False, high_res=True):
        super().__init__()
        self.depth = depth
        self.is_4f = is_4f
        self.high_res = high_res

        self.in_conv = ConvBNReLU(in_ch, out_ch, k=3, p=1)
        self.down_convs = nn.ModuleList()
        self.pools      = nn.ModuleList()
        for _ in range(depth-1):
            self.down_convs.append(ConvBNReLU(out_ch, out_ch, k=3, p=1))
            self.pools.append(nn.MaxPool2d(2,2, ceil_mode=True))

        # bottom/dilated stack
        if is_4f:
            # Atrous pyramid at low resolution (rates 2,4,8) (paper Fig.7).
            self.bottom = nn.Sequential(
                ConvBNReLU(out_ch, out_ch, d=2, p=2),
                ConvBNReLU(out_ch, out_ch, d=4, p=4),
                ConvBNReLU(out_ch, out_ch, d=8, p=8)
            )
        else:
            self.bottom = ConvBNReLU(out_ch, out_ch, k=3, p=1)

        # decoder path with attention gates
        self.up = Up2x()
        self.att_gates = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for _ in range(depth-1):
            self.att_gates.append(AttentionGate(out_ch, out_ch, inter_ch=out_ch))
            self.dec_convs.append(ConvBNReLU(out_ch*2, out_ch, k=3, p=1))

        # Tail: optional CBAM
        if high_res:
            self.tail_att = SpatialAttention(kernel_size=7)  # SAM for high-res
        else:
            self.tail_att = ChannelAttention(out_ch)         # CAM for low-res

    def forward(self, x):
        x0 = self.in_conv(x)

        enc_feats = []
        h = x0
        # encoder with pools
        for i in range(self.depth-1):
            h = self.down_convs[i](h)
            enc_feats.append(h)
            h = self.pools[i](h)

        # bottom
        h = self.bottom(h)

        # decoder with attention-gated skips
        for i in reversed(range(self.depth-1)):
            h = self.up(h, ref=enc_feats[i])
            gated = self.att_gates[i](enc_feats[i], h)
            h = self.dec_convs[i](torch.cat([h, gated], dim=1))

        # tail CBAM
        h = self.tail_att(h)
        return h  # same spatial size as input


# -------------------------
# (DA-U)²Net backbone
# -------------------------

class DAU2Net(nn.Module):
    """
    Six-stage encoder (RSU-7,6,5,4,4F,4F) + five-stage decoder with deep supervision.
    Final outputs: fused logits + side logits s1...s6 (all before sigmoid).
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()

        # Encoder
        self.stage1 = DARSUBlock(in_ch,  64,  64, depth=7, is_4f=False, high_res=True)   # DARSU-7
        self.pool12 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage2 = DARSUBlock( 64, 128, 128, depth=6, is_4f=False, high_res=True)     # DARSU-6
        self.pool23 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage3 = DARSUBlock(128, 256, 256, depth=5, is_4f=False, high_res=True)     # DARSU-5
        self.pool34 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage4 = DARSUBlock(256, 512, 512, depth=4, is_4f=False, high_res=True)     # DARSU-4
        self.pool45 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage5 = DARSUBlock(512, 512, 512, depth=4, is_4f=True, high_res=False)     # DARSU-4F
        self.pool56 = nn.MaxPool2d(2,2, ceil_mode=True)

        self.stage6 = DARSUBlock(512, 512, 512, depth=4, is_4f=True, high_res=False)     # DARSU-4F

        # Decoder (mirror). Each takes cat(previous_dec, encoder_skip)
        self.up = Up2x()

        self.stage6d = DARSUBlock(512, 512, 512, depth=4, is_4f=False, high_res=False)
        self.stage5d = DARSUBlock(512+512, 512, 512, depth=4, is_4f=False, high_res=False)
        self.stage4d = DARSUBlock(512+512, 256, 256, depth=4, is_4f=False, high_res=True)
        self.stage3d = DARSUBlock(256+256, 128, 128, depth=5, is_4f=False, high_res=True)
        self.stage2d = DARSUBlock(128+128, 64,   64,  depth=6, is_4f=False, high_res=True)
        self.stage1d = DARSUBlock(64+64,   64,   64,  depth=7, is_4f=False, high_res=True)

        # Deep supervision: 1x1 conv heads
        self.side1 = nn.Conv2d(64,  out_ch, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(64,  out_ch, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, kernel_size=3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, kernel_size=3, padding=1)

        # Fuse head
        self.out_conv = nn.Conv2d(6*out_ch, out_ch, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]

        # Encoder path
        hx1 = self.stage1(x)
        hx  = self.pool12(hx1)

        hx2 = self.stage2(hx)
        hx  = self.pool23(hx2)

        hx3 = self.stage3(hx)
        hx  = self.pool34(hx3)

        hx4 = self.stage4(hx)
        hx  = self.pool45(hx4)

        hx5 = self.stage5(hx)
        hx  = self.pool56(hx5)

        hx6 = self.stage6(hx)

        # Decoder path (U²Net-style concatenations)
        # stage6d takes hx6
        hd6 = self.stage6d(hx6)                       # -> 512
        hd5 = self.stage5d(torch.cat([self.up(hd6, ref=hx5), hx5], dim=1))    # -> 512
        hd4 = self.stage4d(torch.cat([self.up(hd5, ref=hx4), hx4], dim=1))    # -> 256
        hd3 = self.stage3d(torch.cat([self.up(hd4, ref=hx3), hx3], dim=1))    # -> 128
        hd2 = self.stage2d(torch.cat([self.up(hd3, ref=hx2), hx2], dim=1))    # -> 64
        hd1 = self.stage1d(torch.cat([self.up(hd2, ref=hx1), hx1], dim=1))    # -> 64

        # Side outputs (logits) upsampled to input size
        s1 = F.interpolate(self.side1(hd1), size=(h,w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.side2(hd2), size=(h,w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.side3(hd3), size=(h,w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.side4(hd4), size=(h,w), mode='bilinear', align_corners=False)
        s5 = F.interpolate(self.side5(hd5), size=(h,w), mode='bilinear', align_corners=False)
        s6 = F.interpolate(self.side6(hd6), size=(h,w), mode='bilinear', align_corners=False)

        fuse = self.out_conv(torch.cat([s1, s2, s3, s4, s5, s6], dim=1))  # fused logits

        return fuse, (s1, s2, s3, s4, s5, s6)
