# -*- coding: utf-8 -*-
# PyTorch + TensorFlow (TF1-on-TF2) model components — fixed for checkpoint restore & TF2/DirectML
# Notes:
#   * GAT bias var is now named 'bias' to match checkpoints produced by tf.layers.* (not 'biases')
#   * GN groups use integer math
#   * Infer module uses integer ds_rate keys (1,2,4,8,...) to match cnn_feat dict
#   * No tf.contrib usage anywhere

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small CNN backbone (DRIU‑style lite) ----------
class _Conv(nn.Module):
    def __init__(self, c_in, c_out, ks=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, ks, s, p)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class _VGGBlock(nn.Module):
    def __init__(self, c_in, c_out, n=2):
        super().__init__()
        layers = []
        ch = c_in
        for _ in range(n):
            layers += [_Conv(ch, c_out)]
            ch = c_out
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DRIUBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _VGGBlock( 3,  64, n=2); self.p1 = nn.MaxPool2d(2)
        self.b2 = _VGGBlock(64, 128, n=2); self.p2 = nn.MaxPool2d(2)
        self.b3 = _VGGBlock(128,256, n=3); self.p3 = nn.MaxPool2d(2)
        self.b4 = _VGGBlock(256,512, n=3); self.p4 = nn.MaxPool2d(2)
        self.b5 = _VGGBlock(512,512, n=3)
        self.s1 = nn.Conv2d( 64, 16, 1)
        self.s2 = nn.Conv2d(128, 16, 1)
        self.s3 = nn.Conv2d(256, 16, 1)
        self.s4 = nn.Conv2d(512, 16, 1)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        f1 = self.b1(x)
        f2 = self.b2(self.p1(f1))
        f3 = self.b3(self.p2(f2))
        f4 = self.b4(self.p3(f3))
        f5 = self.b5(self.p4(f4))

        s1 = F.interpolate(self.s1(f1), size=(h,w), mode='bilinear', align_corners=False)
        s2 = F.interpolate(self.s2(f2), size=(h,w), mode='bilinear', align_corners=False)
        s3 = F.interpolate(self.s3(f3), size=(h,w), mode='bilinear', align_corners=False)
        s4 = F.interpolate(self.s4(f4), size=(h,w), mode='bilinear', align_corners=False)
        p_cnn = torch.sigmoid(self.out(torch.cat([s1,s2,s3,s4], dim=1)))

        feats = {"f1": f1, "f2": f2, "f3": f3, "f4": f4, "f5": f5}
        return p_cnn, feats

# ---------- simple GAT ----------
class GATLayer(nn.Module):
    def __init__(self, in_ch, out_ch, heads=4, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.W = nn.Linear(in_ch, heads * out_ch, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_ch))
        self.a_dst = nn.Parameter(torch.empty(heads, out_ch))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, x, edge_index):
        V = x.shape[0]; H = self.heads; Cout = self.W.out_features // H
        z = self.W(x).view(V, H, Cout)
        src, dst = edge_index
        z_src = z[src]; z_dst = z[dst]
        e = (z_src * self.a_src).sum(-1) + (z_dst * self.a_dst).sum(-1)
        e = self.leakyrelu(e)

        out = torch.zeros(V, H, Cout, device=x.device)
        for h in range(H):
            e_h = e[:, h]
            # stable per-dst softmax
            max_per_dst = torch.full((V,), -1e9, device=x.device)
            max_per_dst.scatter_reduce_(0, dst, e_h, reduce='amax', include_self=True)
            exp_e = torch.exp(e_h - max_per_dst[dst])
            denom = torch.zeros(V, device=x.device)
            denom.scatter_add_(0, dst, exp_e)
            alpha = exp_e / (denom[dst] + 1e-9)

            agg = torch.zeros(V, Cout, device=x.device)
            agg.index_add_(0, dst, z_src[:, h, :] * alpha.unsqueeze(-1))
            out[:, h, :] = F.elu(agg)
        return out.reshape(V, H*Cout)

# ---------- helpers shared by train/test ----------
def _align(x, ref_hw):
    if x.shape[-2:] != ref_hw:
        x = F.interpolate(x, size=ref_hw, mode='bilinear', align_corners=False)
    return x

def _scale_verts_to_hw(verts, src_hw, dst_hw):
    Hs, Ws = src_hw; Hd, Wd = dst_hw
    if (Hs, Ws) == (Hd, Wd):
        return verts
    vy = verts[:, 0].float() * (Hd - 1) / max(1.0, (Hs - 1))
    vx = verts[:, 1].float() * (Wd - 1) / max(1.0, (Ws - 1))
    vy = torch.clamp(torch.round(vy), 0, Hd - 1).long()
    vx = torch.clamp(torch.round(vx), 0, Wd - 1).long()
    return torch.stack([vy, vx], dim=1)

def srvs_vertices(prob, delta):
    _, H, W = prob.shape
    cell = 2 * int(delta)
    Hc = math.ceil(H / cell); Wc = math.ceil(W / cell)
    ys, xs = [], []
    with torch.no_grad():
        P = F.pad(prob, (0, Wc*cell - W, 0, Hc*cell - H), mode='replicate')
        P = P.view(1, Hc, cell, Wc, cell).permute(0,1,3,2,4).reshape(1, Hc, Wc, cell*cell)
        idx = P.argmax(dim=-1)
        for iy in range(Hc):
            for ix in range(Wc):
                off = idx[0,iy,ix].item()
                dy, dx = divmod(off, cell)
                y = min(iy*cell + dy, H-1); x = min(ix*cell + dx, W-1)
                ys.append(y); xs.append(x)
    verts = torch.tensor(list(zip(ys,xs)), dtype=torch.long, device=prob.device)
    return verts, (Hc, Wc), cell

def edge_index_euclidean(verts, k=8):
    V = verts.shape[0]
    vy = verts[:,0].float(); vx = verts[:,1].float()
    coords = torch.stack([vy, vx], dim=1)
    d2 = torch.cdist(coords, coords)
    knn = torch.topk(d2, k=k+1, largest=False).indices[:,1:]
    src = torch.arange(V, device=verts.device).unsqueeze(1).repeat(1,k)
    edge_index = torch.stack([src.reshape(-1), knn.reshape(-1)], dim=0)
    return edge_index

def edge_index_geodesic(prob, verts, d_thresh, k_fallback=8):
    H, W = prob.shape[-2:]
    V = verts.shape[0]
    vy = verts[:,0].float(); vx = verts[:,1].float()
    coords = torch.stack([vy, vx], dim=1)

    with torch.no_grad():
        d2 = torch.cdist(coords, coords)
        knn = torch.topk(d2, k=k_fallback+1, largest=False).indices[:,1:]
        edges = []
        P = prob[0]

        def geo_cost(a, b):
            ay, ax = int(a[0]), int(a[1]); by, bx = int(b[0]), int(b[1])
            y0, y1 = sorted([ay, by]); x0, x1 = sorted([ax, bx])
            y0 = max(0, y0-8); y1 = min(H-1, y1+8)
            x0 = max(0, x0-8); x1 = min(W-1, x1+8)
            sub = P[y0:y1+1, x0:x1+1].clone()
            sy, sx = ay-y0, ax-x0; ty, tx = by-y0, bx-x0
            inf = 1e9
            dist = torch.full_like(sub, inf); dist[sy, sx] = 0.0
            vis = torch.zeros_like(sub, dtype=torch.bool)
            for _ in range(sub.numel()):
                cur = torch.where(~vis, dist, torch.tensor(inf, device=sub.device)).view(-1)
                idx = torch.argmin(cur).item()
                cy = idx // sub.shape[1]; cx = idx % sub.shape[1]
                if vis[cy,cx]: break
                vis[cy,cx] = True
                if cy == ty and cx == tx: break
                p0 = sub[cy,cx]
                for dy in (-1,0,1):
                    for dx in (-1,0,1):
                        if dy==0 and dx==0: continue
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < sub.shape[0] and 0 <= nx < sub.shape[1]:
                            w = (p0 - sub[ny,nx]).abs()
                            if dist[cy,cx] + w < dist[ny,nx]:
                                dist[ny,nx] = dist[cy,cx] + w
            return float(dist[ty,tx].item())

        for i in range(V):
            a = coords[i]
            for j in knn[i]:
                b = coords[j]
                gd = geo_cost(a, b)
                if gd <= d_thresh:
                    edges.append([i, int(j.item())])

        if not edges:
            for i in range(V-1):
                edges.append([i, i+1])

        return torch.tensor(edges, dtype=torch.long, device=prob.device).t().contiguous()

# ---------- inference head ----------
class InferenceModule(nn.Module):
    def __init__(self, gnn_ch, cnn_chs=(64,128,256,512,512)):
        super().__init__()
        self.compress = nn.Conv2d(gnn_ch, 64, 1)
        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.fuse4 = _VGGBlock(64+cnn_chs[3], 64, n=2)
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.fuse3 = _VGGBlock(64+cnn_chs[2], 64, n=2)
        self.up2 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.fuse2 = _VGGBlock(64+cnn_chs[1], 64, n=2)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2); self.fuse1 = _VGGBlock(64+cnn_chs[0], 64, n=2)
        self.out  = nn.Conv2d(64, 1, 1)

    def forward(self, gnn_tensor, feats):
        gnn_tensor = _align(gnn_tensor, feats["f5"].shape[-2:])
        x = self.compress(gnn_tensor)
        x = self.up4(x); x = _align(x, feats["f4"].shape[-2:]); x = self.fuse4(torch.cat([x, feats["f4"]], dim=1))
        x = self.up3(x); x = _align(x, feats["f3"].shape[-2:]); x = self.fuse3(torch.cat([x, feats["f3"]], dim=1))
        x = self.up2(x); x = _align(x, feats["f2"].shape[-2:]); x = self.fuse2(torch.cat([x, feats["f2"]], dim=1))
        x = self.up1(x); x = _align(x, feats["f1"].shape[-2:]); x = self.fuse1(torch.cat([x, feats["f1"]], dim=1))
        return torch.sigmoid(self.out(x))

# ---------- VGN ----------
class VGN(nn.Module):
    def __init__(self, gnn_heads=4, gnn_hidden=16):
        super().__init__()
        self.cnn = DRIUBackbone()
        self.gat1 = GATLayer(in_ch=5, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat2 = GATLayer(in_ch=gnn_hidden*gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.gat3 = GATLayer(in_ch=gnn_hidden*gnn_heads, out_ch=gnn_hidden, heads=gnn_heads)
        self.gnn_pred = nn.Linear(gnn_hidden*gnn_heads, 1)
        self.infer = InferenceModule(gnn_ch=gnn_hidden*gnn_heads)

    @torch.no_grad()
    def _tensorize_gnn(self, verts, grid_hw, cell, gnn_feat, ref_feat_spatial):
        Hc, Wc = grid_hw; Cg = gnn_feat.shape[1]
        grid = torch.zeros(1, Cg, Hc, Wc, device=gnn_feat.device)
        ys = torch.clamp(verts[:,0] // cell, 0, Hc-1)
        xs = torch.clamp(verts[:,1] // cell, 0, Wc-1)
        grid[0, :, ys, xs] = gnn_feat.T
        return F.interpolate(grid, size=ref_feat_spatial, mode='bilinear', align_corners=False)

    def forward(self, x, cache_graph=None, delta=4, geo_thresh=10, edge_mode="geodesic"):
        assert x.shape[0] == 1, "VGN assumes batch=1."
        p_cnn, feats = self.cnn(x)
        H, W = p_cnn.shape[-2:]
        prob = p_cnn.detach()

        # graph
        if cache_graph is None:
            verts, grid_hw, cell = srvs_vertices(prob[0], delta)
            if edge_mode == "geodesic":
                edges = edge_index_geodesic(prob[0], verts, geo_thresh)
            else:
                edges = edge_index_euclidean(verts)
            cache_graph = {"verts": verts, "edges": edges, "grid_hw": grid_hw, "cell": cell, "src_hw": (H, W)}

        verts_raw = cache_graph["verts"]
        src_hw = cache_graph.get("src_hw", (H, W))
        verts = _scale_verts_to_hw(verts_raw, src_hw, (H, W))

        cell = 2 * int(delta)
        grid_hw = (math.ceil(H / cell), math.ceil(W / cell))
        edges = cache_graph["edges"]

        with torch.no_grad():
            y  = (verts[:,0].float() / max(1, (H-1))).unsqueeze(1)
            xk = (verts[:,1].float() / max(1, (W-1))).unsqueeze(1)
            p  = prob[0, 0, verts[:,0], verts[:,1]].unsqueeze(1)
            g  = x[0, 1, verts[:,0], verts[:,1]].unsqueeze(1)
            node_feat = torch.cat([y, xk, p, g, 1.0 - p], dim=1)

        h1 = self.gat1(node_feat, edges)
        h2 = self.gat2(h1, edges)
        h3 = self.gat3(h2, edges)
        p_gnn = torch.sigmoid(self.gnn_pred(h3)).squeeze(1)

        gnn_tensor = self._tensorize_gnn(verts, grid_hw, cell, h3, feats["f5"].shape[-2:])
        p_vgn = self.infer(gnn_tensor, feats)
        graph_snapshot = {"verts": verts_raw, "edges": edges, "grid_hw": grid_hw, "cell": cell, "src_hw": src_hw}
        return {"p_cnn": p_cnn, "p_vgn": p_vgn, "p_gnn": p_gnn, "graph": graph_snapshot}


# =====================================================================
# =================== TensorFlow (TF1-on-TF2) Part ====================
# =====================================================================

# -*- coding: utf-8 -*-
""" Common model file — TF2/DirectML compatible header & helpers (fixed) """

import numpy as np
import pickle
try:
    import tensorflow as tf
except ImportError:
    tf = None


class _TensorFlowStub:
    def __getattr__(self, _name):
        raise ImportError("TensorFlow is required for this functionality.")


TF_AVAILABLE = tf is not None

# ---- TF1 compatibility for TF2 runtimes (e.g., DirectML wheels) -------------
if TF_AVAILABLE and hasattr(tf, "compat") and hasattr(tf.compat, "v1"):
    tf = tf.compat.v1  # rebind for readability
    try:
        tf.disable_v2_behavior()
    except Exception:
        try:
            tf.disable_eager_execution()
        except Exception:
            pass
    # Aliases commonly used by old code
    tf.placeholder = tf.compat.v1.placeholder
    tf.sparse_placeholder = tf.compat.v1.sparse_placeholder
    tf.variable_scope = tf.compat.v1.variable_scope
    tf.get_variable = tf.compat.v1.get_variable
    tf.Session = tf.compat.v1.Session
    tf.InteractiveSession = tf.compat.v1.InteractiveSession
    tf.ConfigProto = tf.compat.v1.ConfigProto
    tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
    tf.summary = tf.compat.v1.summary
    tf.layers = tf.compat.v1.layers
    tf.train = tf.compat.v1.train
    tf.placeholder_with_default = tf.compat.v1.placeholder_with_default
    if not hasattr(tf, "count_nonzero"): tf.count_nonzero = tf.math.count_nonzero
    if not hasattr(tf, "to_float"): tf.to_float = lambda x, name=None: tf.cast(x, tf.float32, name=name)
    if not hasattr(tf, "trainable_variables"): tf.trainable_variables = tf.compat.v1.trainable_variables
    if not hasattr(tf, "sparse_add"):
        try: tf.sparse_add = tf.compat.v1.sparse_add
        except AttributeError:
            def _sparse_add(a, b, name=None): return tf.sparse.add(a, b, name=name)
            tf.sparse_add = _sparse_add
    if not hasattr(tf, "sparse_softmax"):
        try: tf.sparse_softmax = tf.compat.v1.sparse_softmax
        except AttributeError:
            def _sparse_softmax(sp_input, name=None): return tf.sparse.softmax(sp_input, name=name)
            tf.sparse_softmax = _sparse_softmax
    if not hasattr(tf, "sparse_reshape"):
        try: tf.sparse_reshape = tf.compat.v1.sparse_reshape
        except AttributeError:
            def _sparse_reshape(sp_input, shape, name=None): return tf.sparse.reshape(sp_input, shape, name=name)
            tf.sparse_reshape = _sparse_reshape
    if not hasattr(tf, "sparse_tensor_dense_matmul"):
        try: tf.sparse_tensor_dense_matmul = tf.compat.v1.sparse_tensor_dense_matmul
        except AttributeError:
            def _sparse_tensor_dense_matmul(sp_a, b, adjoint_a=False, adjoint_b=False, name=None):
                return tf.sparse.sparse_dense_matmul(sp_a, b, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name=name)
            tf.sparse_tensor_dense_matmul = _sparse_tensor_dense_matmul
elif not TF_AVAILABLE:
    tf = _TensorFlowStub()
# ensure helper sees lack of TF
TF_AVAILABLE = TF_AVAILABLE and not isinstance(tf, _TensorFlowStub)

from config import cfg

DEBUG = False

# ---- Tiny alias helper
def _alias(name, fn):
    if not TF_AVAILABLE:
        return
    if fn is None: 
        return
    if not hasattr(tf, name):
        setattr(tf, name, fn)

# Basic math aliases used by legacy code
_alias('ceil', tf.math.ceil)
_alias('floor', tf.math.floor)
_alias('round', tf.math.round)
_alias('log', tf.math.log)
_alias('exp', tf.math.exp)
_alias('rsqrt', tf.math.rsqrt)
_alias('div', tf.math.divide)
_alias('truediv', tf.math.truediv)
_alias('to_float', lambda x, name=None: tf.cast(x, tf.float32, name=name))
_alias('to_int32', lambda x, name=None: tf.cast(x, tf.int32, name=name))

# ---- Initializer shims (safe for TF2/DirectML)
def _get_truncated_normal_initializer(mean=0.0, stddev=0.01, seed=None, dtype=tf.float32):
    try:
        return tf.compat.v1.truncated_normal_initializer(mean=mean, stddev=stddev, seed=seed, dtype=dtype)
    except Exception:
        try:
            return tf.keras.initializers.TruncatedNormal(mean=mean, stddev=stddev, seed=seed)
        except Exception:
            def _init(shape, dtype=None, partition_info=None):
                return tf.random.truncated_normal(shape, mean=mean, stddev=stddev, dtype=dtype or tf.float32, seed=seed)
            return _init

def _get_glorot_uniform_initializer():
    try:
        return tf.compat.v1.glorot_uniform_initializer()
    except Exception:
        try:
            return tf.keras.initializers.GlorotUniform()
        except Exception:
            return tf.initializers.random_uniform(minval=-0.05, maxval=0.05)

def _activation_summary(x_name, x):
    tf.summary.histogram(x_name + '/activations', x)
    tf.summary.scalar(x_name + '/sparsity', tf.nn.zero_fraction(x))


# https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
def get_deconv_filter(f_shape):
    width = f_shape[0]; height = f_shape[0]
    f = np.ceil(width/2.0); c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(height):
            bilinear[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    return weights


def add_tensors_wo_none(tensor_list):
    # Adds all input tensors element-wise while filtering out none tensors.
    temp_list = [x for x in tensor_list if (x is not None)]
    if len(temp_list):
        # Handle IndexedSlices if present
        if any(isinstance(x, tf.IndexedSlices) for x in temp_list):
            temp_list = [tf.convert_to_tensor(x) if isinstance(x, tf.IndexedSlices) else x for x in temp_list]
        return tf.add_n(temp_list)
    else:
        return None
     

class base_model():
    def __init__(self, weight_file_path):
        if weight_file_path is not None:
            with open(weight_file_path, 'rb') as f:
                self.pretrained_weights = pickle.load(f)       
    
    def new_conv_layer(self, bottom, filter_shape, stride=[1,1,1,1], init=None, \
                       norm_type=None, use_relu=False, is_training=True, name=None):
        if init is None:
            init = _get_truncated_normal_initializer(0., 0.01)
        with tf.variable_scope(name) as scope:
            w = tf.get_variable("W", shape=filter_shape, initializer=init)
            out = tf.nn.conv2d(bottom, w, stride, padding='SAME')
            
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                num_groups = max(1, int(filter_shape[-1] // max(1, cfg.GN_MIN_CHS_PER_G)))
                out = self.group_norm(out, num_group=min(cfg.GN_MIN_NUM_G, num_groups))
            else:
                b = tf.get_variable("b", shape=filter_shape[-1], initializer=tf.constant_initializer(0.))
                out = tf.nn.bias_add(out, b)

            if use_relu:
                out = tf.nn.relu(out)

        return out
    
    def new_fc_layer(self, bottom, input_size, output_size, init=None, \
                     norm_type=None, use_relu=False, is_training=True, name=None):
        if init is None:
            init = _get_truncated_normal_initializer(0., 0.01)
        shape = bottom.get_shape().as_list()
        dim = np.prod(shape[1:])
        x = tf.reshape(bottom, [-1,dim])

        with tf.variable_scope(name) as scope:
            w = tf.get_variable("W", shape=[input_size, output_size], initializer=init)
            out = tf.matmul(x, w)
            
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                num_groups = max(1, int(output_size // max(1, cfg.GN_MIN_CHS_PER_G)))
                out = self.group_norm_fc(out, num_group=min(cfg.GN_MIN_NUM_G, num_groups))
            else:
                b = tf.get_variable("b", shape=[output_size], initializer=tf.constant_initializer(0.))
                out = out+b
                
            if use_relu:
                out = tf.nn.relu(out)

        return out
    
    def new_deconv_layer(self, bottom, filter_shape, output_shape, strides, norm_type=None, use_relu=False, is_training=True, name=None):
        weights = get_deconv_filter(filter_shape)       
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                "W",
                shape=weights.shape,
                dtype=tf.float32,
                initializer=tf.compat.v1.constant_initializer(value=weights, dtype=tf.float32)
            )

            out = tf.nn.conv2d_transpose(bottom, w, output_shape, strides, padding='SAME')
            
            if DEBUG:
                out = tf.Print(out, [tf.shape(out)],
                               message='Shape of %s' % name,
                               summarize=4, first_n=1)
                
            if norm_type=='BN':
                out = tf.layers.batch_normalization(out, training=is_training, renorm=cfg.USE_BRN)
            elif norm_type=='GN':
                num_groups = max(1, int(filter_shape[-2] // max(1, cfg.GN_MIN_CHS_PER_G)))
                out = self.group_norm(out, num_group=min(cfg.GN_MIN_NUM_G, num_groups))
            else:
                b = tf.get_variable("b", shape=weights.shape[-2], initializer=tf.constant_initializer(0.))
                out = tf.nn.bias_add(out, b)

            if use_relu:
                out = tf.nn.relu(out)    

        return out

    # https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
    # https://github.com/tensorflow/tensorflow/issues/2169        
    def unpool(self, pool, ind, ksize, name):
        with tf.variable_scope(name) as scope:
            input_shape =  tf.shape(pool)
            output_shape = [input_shape[0], input_shape[1]*ksize[1], input_shape[2]*ksize[2], input_shape[3]]
    
            flat_input_size = tf.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]])
    
            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                              shape=tf.stack([input_shape[0], 1, 1, 1]))
            b = tf.ones_like(ind)*batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)
    
            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))
        
        return ret
    
    def group_norm(self, input, num_group=32, epsilon=1e-05):
        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NHWCG)
        mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
        beta  = tf.get_variable('beta',  [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))
        output = tf.reshape(output, tf.shape(input)) * gamma + beta
        return output
    
    def group_norm_fc(self, input, num_group=32, epsilon=1e-05):
        num_ch = input.get_shape().as_list()[-1]
        num_group = min(num_group, num_ch)
        NCG = tf.concat([tf.slice(tf.shape(input),[0],[1]), tf.constant([num_ch//num_group, num_group])], axis=0)
        output = tf.reshape(input, NCG)
        mean, var = tf.nn.moments(output, [1], keep_dims=True)
        output = (output - mean) / tf.sqrt(var + epsilon)
        gamma = tf.get_variable('gamma', [1, num_ch], initializer=tf.constant_initializer(1.0))
        beta  = tf.get_variable('beta',  [1, num_ch], initializer=tf.constant_initializer(0.0))
        output = tf.reshape(output, tf.shape(input)) * gamma + beta
        return output
    
    def group_norm_layer(self, input, num_group=32, epsilon=1e-05, name=None):
        with tf.variable_scope(name) as scope:
            num_ch = input.get_shape().as_list()[-1]
            num_group = min(num_group, num_ch)
            NHWCG = tf.concat([tf.slice(tf.shape(input),[0],[3]), tf.constant([num_ch//num_group, num_group])], axis=0)
            output = tf.reshape(input, NHWCG)
            mean, var = tf.nn.moments(output, [1, 2, 3], keep_dims=True)
            output = (output - mean) / tf.sqrt(var + epsilon)
            gamma = tf.get_variable('gamma', [1, 1, 1, num_ch], initializer=tf.constant_initializer(1.0))
            beta  = tf.get_variable('beta',  [1, 1, 1, num_ch], initializer=tf.constant_initializer(0.0))
            output = tf.reshape(output, tf.shape(input)) * gamma + beta
        return output


class vessel_segm_cnn(base_model):
    def __init__(self, params, weight_file_path):
        base_model.__init__(self, weight_file_path)
        self.params = params
        self.cnn_model = params.cnn_model
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build_model()
        
    def build_model(self):
        print("Building the model...")
        if self.cnn_model=='driu':
            self.build_driu()
        elif self.cnn_model=='driu_large':
            self.build_driu_large()
        print("Model built.")
        
    def build_driu(self):
        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')
        is_training = tf.placeholder(tf.bool, [])

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)

        # specialized layers        
        num_ch = 16
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]), tf.constant(num_ch, shape=[1])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,num_ch], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,num_ch], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,num_ch,num_ch], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,num_ch], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,num_ch,num_ch], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,num_ch], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,num_ch,num_ch], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_concat = tf.concat(values=[spe_1, resized_spe_2, resized_spe_3, resized_spe_4], axis=3)

        output = self.new_conv_layer(spe_concat, [1,1,num_ch*4,1], name='output')
        _activation_summary('output', output)
        
        fg_prob = tf.sigmoid(output)

        # --- Weighted cross entropy loss (in FOV) ---
        binary_mask_fg = tf.cast(tf.equal(labels, 1), tf.float32)
        binary_mask_bg = tf.cast(tf.not_equal(labels, 1), tf.float32)
        combined_mask = tf.concat(values=[binary_mask_bg, binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=output, shape=(-1,))        
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.cast(flat_labels, tf.float32))
        
        fov_f = tf.cast(fov_masks, tf.float32) * tf.ones_like(labels, dtype=tf.float32)
        num_pixel     = tf.reduce_sum(fov_f) + cfg.EPSILON
        num_pixel_fg  = tf.reduce_sum(binary_mask_fg * fov_f)
        num_pixel_bg  = num_pixel - num_pixel_fg

        class_weight_vec = tf.stack([num_pixel_fg / num_pixel, num_pixel_bg / num_pixel])  # [2]
        weight_per_label = tf.reduce_sum(flat_one_hot_labels * class_weight_vec[tf.newaxis, :], axis=1)

        reshaped_fov_masks = tf.reshape(fov_f, (-1,))
        reshaped_fov_masks /= (tf.reduce_mean(reshaped_fov_masks) + cfg.EPSILON)

        loss = tf.reduce_mean(cross_entropies * weight_per_label * reshaped_fov_masks)
         
        weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        # --- Metrics ---
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=fg_prob, shape=(-1,)), 0.5)
        correct = tf.cast(tf.equal(flat_bin_output, tf.cast(flat_labels, tf.bool)), tf.float32)
        accuracy = tf.reduce_mean(correct)
        num_fg_output = tf.reduce_sum(tf.cast(flat_bin_output, tf.float32))
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(flat_labels, tf.bool), flat_bin_output), tf.float32))
        pre = tf.divide(tp, tf.add(num_fg_output, cfg.EPSILON))
        rec = tf.divide(tp, tf.cast(num_pixel_fg, tf.float32) + cfg.EPSILON)
        
        # --- Optimizer ---
        if self.params.opt=='adam':
            train_op = tf.train.AdamOptimizer(self.params.lr, epsilon=0.1).minimize(loss, global_step=self.global_step)
        elif self.params.opt=='sgd':
            if self.params.lr_decay=='const':
                opt = tf.train.MomentumOptimizer(self.params.lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), gvs)
                gvs = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), gvs)
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)
            elif self.params.lr_decay=='pc':
                boundaries = [int(self.params.max_iters*0.5), int(self.params.max_iters*0.75)]
                values = [self.params.lr, self.params.lr*0.5, self.params.lr*0.25]
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
                opt = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), gvs)
                gvs = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), gvs)
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)
            elif self.params.lr_decay=='exp':
                lr = tf.train.exponential_decay(self.params.lr, self.global_step, self.params.max_iters/20, 0.9, staircase=False)
                opt = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in gvs]
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)

        # Hang up
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training
        self.conv_feats = spe_concat
        self.output = output
        self.fg_prob = fg_prob
        self.flat_bin_output = flat_bin_output
        self.tp = tp
        self.num_fg_output = num_fg_output
        self.num_pixel_fg = num_pixel_fg        
        self.accuracy = accuracy
        self.precision = pre        
        self.recall = rec
        self.loss = loss
        self.train_op = train_op

        
    def build_driu_large(self):
        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')
        is_training = tf.placeholder(tf.bool, [])

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)
        pool4= tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        
        conv5_1 = self.new_conv_layer(pool4, [3,3,512,512], use_relu=True, name='conv5_1')
        _activation_summary('conv5_1', conv5_1)
        conv5_2 = self.new_conv_layer(conv5_1, [3,3,512,512], use_relu=True, name='conv5_2')
        _activation_summary('conv5_2', conv5_2)
        conv5_3 = self.new_conv_layer(conv5_2, [3,3,512,512], use_relu=True, name='conv5_3')
        _activation_summary('conv5_3', conv5_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]), tf.constant(16, shape=[1])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2')
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')
        spe_5 = self.new_conv_layer(conv5_3, [3,3,512,16], use_relu=True, name='spe_5')
        _activation_summary('spe_5', spe_5)
        resized_spe_5 = self.new_deconv_layer(spe_5, [32,32,16,16], target_shape, [1,16,16,1], use_relu=True, name='resized_spe_5')
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4,resized_spe_5], axis=3)

        output = self.new_conv_layer(spe_concat, [1,1,16*5,1], name='output')
        _activation_summary('output', output)
        
        fg_prob = tf.sigmoid(output)

        # weighted cross entropy loss
        binary_mask_fg = tf.cast(tf.equal(labels, 1), tf.float32)
        binary_mask_bg = tf.cast(tf.not_equal(labels, 1), tf.float32)
        combined_mask = tf.concat(values=[binary_mask_bg,binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=output, shape=(-1,))        
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.cast(flat_labels, tf.float32))
        
        num_pixel = tf.size(labels)
        num_pixel_fg = tf.count_nonzero(binary_mask_fg, dtype=tf.int32)
        num_pixel_bg = num_pixel - num_pixel_fg
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(num_pixel_fg,num_pixel),(1,1)),
                                                tf.reshape(tf.divide(num_pixel_bg,num_pixel),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels,tf.transpose(class_weight)))
        loss = tf.reduce_mean(tf.multiply(weight_per_label,cross_entropies))
         
        weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=fg_prob, shape=(-1,)), 0.5)
        correct = tf.cast(tf.equal(flat_bin_output,tf.cast(flat_labels, tf.bool)), tf.float32)
        accuracy = tf.reduce_mean(correct)
        num_fg_output = tf.reduce_sum(tf.cast(flat_bin_output, tf.float32))
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), flat_bin_output), tf.float32))
        pre = tf.divide(tp,tf.add(num_fg_output,cfg.EPSILON))
        rec = tf.divide(tp, tf.cast(num_pixel_fg, tf.float32))
        
        if self.params.opt=='adam':
            train_op = tf.train.AdamOptimizer(self.params.lr, epsilon=0.1).minimize(loss, global_step=self.global_step)
        elif self.params.opt=='sgd':
            if self.params.lr_decay=='const':
                opt = tf.train.MomentumOptimizer(self.params.lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), gvs)
                gvs = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), gvs)
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)
            elif self.params.lr_decay=='pc':
                boundaries = [int(self.params.max_iters*0.5), int(self.params.max_iters*0.75)]
                values = [self.params.lr,self.params.lr*0.5,self.params.lr*0.25]
                lr = tf.train.piecewise_constant(self.global_step, boundaries, values)
                opt = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = map(lambda gv: (0.01*gv[0],gv[1]) if 'output' in gv[1].name else (gv[0],gv[1]), gvs)
                gvs = map(lambda gv: (None,gv[1]) if 'resized' in gv[1].name else (tf.clip_by_value(gv[0], -5., 5.),gv[1]), gvs)
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)
            elif self.params.lr_decay=='exp':
                lr = tf.train.exponential_decay(self.params.lr, self.global_step, self.params.max_iters/20, 0.9, staircase=False)
                opt = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM, use_nesterov=True)
                gvs = opt.compute_gradients(loss)
                gvs = [(tf.clip_by_value(gv[0], -5., 5.), gv[1]) for gv in gvs]
                train_op = opt.apply_gradients(gvs, global_step=self.global_step)

        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training
        self.conv_feats = spe_concat
        self.output = output
        self.fg_prob = fg_prob
        self.flat_bin_output = flat_bin_output
        self.tp = tp
        self.num_fg_output = num_fg_output
        self.num_pixel_fg = num_pixel_fg        
        self.accuracy = accuracy
        self.precision = pre        
        self.recall = rec
        self.loss = loss
        self.train_op = train_op

        
class vessel_segm_vgn(base_model):
    def __init__(self, params, weight_file_path):
        base_model.__init__(self, weight_file_path)
        self.params = params
        
        # feed external CNN probability map instead of running the TF CNN
        self.use_external_prob = bool(getattr(params, 'use_external_prob', False) or
                                    getattr(params, 'use_external_cnn_prob', False) or
                                    getattr(params, 'use_external_cnn_probs', False))
        
        # cnn module related
        self.cnn_model = params.cnn_model
        self.cnn_loss_on = params.cnn_loss_on
        
        # gnn module related
        self.win_size = params.win_size
        self.gnn_loss_on = params.gnn_loss_on
        self.gnn_loss_weight = params.gnn_loss_weight
        
        # inference module related
        self.infer_module_kernel_size = params.infer_module_kernel_size
        # legacy checkpoints expect this flag; default to False when missing
        self.use_enc_layer = getattr(params, 'use_enc_layer', False)
        if not hasattr(self.params, 'infer_module_grad_weight'):
            self.params.infer_module_grad_weight = 1.0

        # make global_step explicitly int32 so schedulers/comparisons stay type-safe
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        
        self.build_model()
        
    # special layer for graph attention network
    def sp_attn_head(self, bottom, output_size, adj, name, act=tf.nn.elu, feat_dropout=0., att_dropout=0., residual=False, show_adj=False):
        with tf.variable_scope(name) as scope:
            if feat_dropout != 0.0:
                bottom = tf.nn.dropout(bottom, 1.0 - feat_dropout)

            fts = tf.layers.conv1d(bottom, output_size, 1, use_bias=False)

            # simplest self-attention possible
            f_1 = tf.layers.conv1d(fts, 1, 1)
            f_2 = tf.layers.conv1d(fts, 1, 1)
            
            num_nodes = tf.slice(tf.shape(adj), [0], [1])
            f_1 = tf.reshape(f_1, tf.concat([num_nodes, tf.constant([1])], axis=0))
            f_2 = tf.reshape(f_2, tf.concat([num_nodes, tf.constant([1])], axis=0)) 

            f_1 = adj * f_1
            f_2 = adj * tf.transpose(f_2, [1,0])

            logits = tf.sparse_add(f_1, f_2)
            lrelu = tf.SparseTensor(indices=logits.indices, 
                                    values=tf.nn.leaky_relu(logits.values),
                                    dense_shape=logits.dense_shape)
            coefs = tf.sparse_softmax(lrelu)

            if att_dropout != 0.0:
                coefs = tf.SparseTensor(indices=coefs.indices,
                                        values=tf.nn.dropout(coefs.values, 1.0 - att_dropout),
                                        dense_shape=coefs.dense_shape)
            if feat_dropout != 0.0:
                fts = tf.nn.dropout(fts, 1.0 - feat_dropout)

            coefs = tf.sparse_reshape(coefs, tf.concat([num_nodes, num_nodes], axis=0))
            fts = tf.squeeze(fts, [0])
            vals = tf.sparse_tensor_dense_matmul(coefs, fts)
            vals = tf.expand_dims(vals, axis=0)
            vals = tf.reshape(vals, tf.concat([tf.constant([1]), num_nodes, tf.constant([output_size])], axis=0))

            # Explicit bias variable named 'bias' (matches checkpoints produced by tf.layers)
            b = tf.get_variable('bias', shape=[output_size], initializer=tf.zeros_initializer())
            ret = vals + b  # broadcast add last dim
    
            # residual connection
            if residual:
                if bottom.shape[-1] != ret.shape[-1]:
                    ret = ret + tf.layers.conv1d(bottom, int(ret.shape[-1]), 1)
                else:
                    ret = ret + bottom

        return (act(ret), coefs) if show_adj else act(ret)
    
    def build_model(self):
        print("Building the model...")
        self.build_cnn_module()
        self.build_gat()  # GAT for our GNN module
        self.build_infer_module()
        self.build_optimizer()
        print("Model built.")
        
    def build_cnn_module(self):
        print("Building the CNN module...")
        if getattr(self, 'use_external_prob', False):
            self.build_external_prob_backbone()    # << use DAU2Net prob only
        else:
            if self.cnn_model=='driu':
                self.build_driu()
            elif self.cnn_model=='driu_large':
                self.build_driu_large()
            else:
                raise NotImplementedError
        print("CNN module built.")
        
    def build_external_prob_backbone(self):
        """
        Minimal 'CNN' that takes an EXTERNAL foreground probability (from DAU2Net)
        and synthesizes the multiscale tensors (spe_1..spe_4) the GNN+infer module expect.
        """
        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')  # still used for shapes/normalization
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')
        is_training = tf.placeholder(tf.bool, [], name='is_training')

        # === external prob from DAU2Net ===
        self.external_cnn_prob = tf.placeholder(tf.float32, [None, None, None, 1], name='external_cnn_prob')
        eps = 1e-7
        ext = tf.clip_by_value(self.external_cnn_prob, eps, 1.0 - eps)

        # --- multi-scale tensors derived from the prob map ---
        target_shape = tf.concat([tf.slice(tf.shape(imgs), [0], [3]), tf.constant(16, shape=[1])], axis=0)

        down2 = tf.nn.avg_pool(ext,   ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='ext_down2')
        down4 = tf.nn.avg_pool(down2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='ext_down4')
        down8 = tf.nn.avg_pool(down4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='ext_down8')

        # produce 16‑ch features per scale (learnable 3×3 conv)
        spe_1 = self.new_conv_layer(ext,   [3,3,1,16], use_relu=True, name='spe_1')
        spe_2 = self.new_conv_layer(down2, [3,3,1,16], use_relu=True, name='spe_2')
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2')

        spe_3 = self.new_conv_layer(down4, [3,3,1,16], use_relu=True, name='spe_3')
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')

        spe_4 = self.new_conv_layer(down8, [3,3,1,16], use_relu=True, name='spe_4')
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')

        # full-res conv feature map used by the GNN gather
        spe_concat = tf.concat([spe_1, resized_spe_2, resized_spe_3, resized_spe_4], axis=3)

        # treat external prob as CNN output (logits needed for cnn_loss / metrics)
        img_output  = tf.log(ext) - tf.log(1.0 - ext)
        img_fg_prob = ext

        # --- Hang up (API-compatible attributes) ---
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training

        self.cnn_feat = {1: spe_1, 2: spe_2, 4: spe_3, 8: spe_4}
        self.cnn_feat_spatial_sizes = {
            1: tf.slice(tf.shape(spe_1), [1], [2]),
            2: tf.slice(tf.shape(spe_2), [1], [2]),
            4: tf.slice(tf.shape(spe_3), [1], [2]),
            8: tf.slice(tf.shape(spe_4), [1], [2]),
        }
        self.conv_feats  = spe_concat
        self.img_output  = img_output
        self.img_fg_prob = img_fg_prob

        # keep scope names compatible with older checkpoints where possible
        self.var_to_restore = [
            'spe_1','spe_2','spe_3','spe_4',
            'resized_spe_2','resized_spe_3','resized_spe_4'
        ]

        
    def build_driu(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs') # RGB
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')
        is_training = tf.placeholder(tf.bool, [], name='is_training')

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]), tf.constant(16, shape=[1])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4], axis=3)
        
        img_output = self.new_conv_layer(spe_concat, [1,1,16*4,1], name='img_output')
        _activation_summary('img_output', img_output)
        
        img_fg_prob = tf.sigmoid(img_output)
        
        # Hang up
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training
        
        self.cnn_feat = {1: spe_1, 2: spe_2, 4: spe_3, 8: spe_4}
        self.cnn_feat_spatial_sizes = {
            1: tf.slice(tf.shape(spe_1),[1],[2]),
            2: tf.slice(tf.shape(spe_2),[1],[2]),
            4: tf.slice(tf.shape(spe_3),[1],[2]),
            8: tf.slice(tf.shape(spe_4),[1],[2]),
        }
        self.conv_feats = spe_concat
        self.img_output = img_output
        self.img_fg_prob = img_fg_prob
        
        self.var_to_restore = [
            'conv1_1','conv1_2',
            'conv2_1','conv2_2',
            'conv3_1','conv3_2','conv3_3',
            'conv4_1','conv4_2','conv4_3',
            'spe_1','spe_2','spe_3','spe_4',
            'resized_spe_2','resized_spe_3','resized_spe_4',
            'img_output'
        ]
        
    def build_driu_large(self):

        imgs = tf.placeholder(tf.float32, [None, None, None, 3], name='imgs')
        labels = tf.placeholder(tf.int64, [None, None, None, 1], name='labels')
        fov_masks = tf.placeholder(tf.int64, [None, None, None, 1], name='fov_masks')
        is_training = tf.placeholder(tf.bool, [], name='is_training')

        conv1_1 = self.new_conv_layer(imgs, [3,3,3,64], use_relu=True, name='conv1_1')
        _activation_summary('conv1_1', conv1_1)
        conv1_2 = self.new_conv_layer(conv1_1, [3,3,64,64], use_relu=True, name='conv1_2')
        _activation_summary('conv1_2', conv1_2)
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
                               
        conv2_1 = self.new_conv_layer(pool1, [3,3,64,128], use_relu=True, name='conv2_1')
        _activation_summary('conv2_1', conv2_1)
        conv2_2 = self.new_conv_layer(conv2_1, [3,3,128,128], use_relu=True, name='conv2_2')
        _activation_summary('conv2_2', conv2_2)
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')                               
                               
        conv3_1 = self.new_conv_layer(pool2, [3,3,128,256], use_relu=True, name='conv3_1')
        _activation_summary('conv3_1', conv3_1)
        conv3_2 = self.new_conv_layer(conv3_1, [3,3,256,256], use_relu=True, name='conv3_2')
        _activation_summary('conv3_2', conv3_2)
        conv3_3 = self.new_conv_layer(conv3_2, [3,3,256,256], use_relu=True, name='conv3_3')
        _activation_summary('conv3_3', conv3_3)
        pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
                               
        conv4_1 = self.new_conv_layer(pool3, [3,3,256,512], use_relu=True, name='conv4_1')
        _activation_summary('conv4_1', conv4_1)
        conv4_2 = self.new_conv_layer(conv4_1, [3,3,512,512], use_relu=True, name='conv4_2')
        _activation_summary('conv4_2', conv4_2)
        conv4_3 = self.new_conv_layer(conv4_2, [3,3,512,512], use_relu=True, name='conv4_3')
        _activation_summary('conv4_3', conv4_3)
        pool4= tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        
        conv5_1 = self.new_conv_layer(pool4, [3,3,512,512], use_relu=True, name='conv5_1')
        _activation_summary('conv5_1', conv5_1)
        conv5_2 = self.new_conv_layer(conv5_1, [3,3,512,512], use_relu=True, name='conv5_2')
        _activation_summary('conv5_2', conv5_2)
        conv5_3 = self.new_conv_layer(conv5_2, [3,3,512,512], use_relu=True, name='conv5_3')
        _activation_summary('conv5_3', conv5_3)

        # specialized layers
        target_shape = tf.concat(values=[tf.slice(tf.shape(imgs), [0], [3]), tf.constant(16, shape=[1])], axis=0)  
        spe_1 = self.new_conv_layer(conv1_2, [3,3,64,16], use_relu=True, name='spe_1')
        _activation_summary('spe_1', spe_1)
        spe_2 = self.new_conv_layer(conv2_2, [3,3,128,16], use_relu=True, name='spe_2')
        _activation_summary('spe_2', spe_2)
        resized_spe_2 = self.new_deconv_layer(spe_2, [4,4,16,16], target_shape, [1,2,2,1], use_relu=True, name='resized_spe_2') 
        spe_3 = self.new_conv_layer(conv3_3, [3,3,256,16], use_relu=True, name='spe_3')
        _activation_summary('spe_3', spe_3)
        resized_spe_3 = self.new_deconv_layer(spe_3, [8,8,16,16], target_shape, [1,4,4,1], use_relu=True, name='resized_spe_3')
        spe_4 = self.new_conv_layer(conv4_3, [3,3,512,16], use_relu=True, name='spe_4')
        _activation_summary('spe_4', spe_4)
        resized_spe_4 = self.new_deconv_layer(spe_4, [16,16,16,16], target_shape, [1,8,8,1], use_relu=True, name='resized_spe_4')  
        spe_5 = self.new_conv_layer(conv5_3, [3,3,512,16], use_relu=True, name='spe_5')
        _activation_summary('spe_5', spe_5)
        resized_spe_5 = self.new_deconv_layer(spe_5, [32,32,16,16], target_shape, [1,16,16,1], use_relu=True, name='resized_spe_5')
        spe_concat = tf.concat(values=[spe_1,resized_spe_2,resized_spe_3,resized_spe_4,resized_spe_5], axis=3)
        
        img_output = self.new_conv_layer(spe_concat, [1,1,16*5,1], name='img_output')
        _activation_summary('img_output', img_output)
        
        img_fg_prob = tf.sigmoid(img_output)
        
        # Hang up
        self.imgs = imgs
        self.labels = labels
        self.fov_masks = fov_masks
        self.is_training = is_training
        
        self.cnn_feat = {1: spe_1, 2: spe_2, 4: spe_3, 8: spe_4, 16: spe_5}
        self.cnn_feat_spatial_sizes = {
            1: tf.slice(tf.shape(spe_1),[1],[2]),
            2: tf.slice(tf.shape(spe_2),[1],[2]),
            4: tf.slice(tf.shape(spe_3),[1],[2]),
            8: tf.slice(tf.shape(spe_4),[1],[2]),
            16: tf.slice(tf.shape(spe_5),[1],[2]),
        }
        self.conv_feats = spe_concat
        self.img_output = img_output
        self.img_fg_prob = img_fg_prob
        
        self.var_to_restore = [
            'conv1_1','conv1_2',
            'conv2_1','conv2_2',
            'conv3_1','conv3_2','conv3_3',
            'conv4_1','conv4_2','conv4_3',
            'conv5_1','conv5_2','conv5_3',
            'spe_1','spe_2','spe_3','spe_4','spe_5',
            'resized_spe_2','resized_spe_3','resized_spe_4','resized_spe_5',
            'img_output'
        ]
        
    def build_gat(self):
        print("Building the GAT part...")
        node_byxs = tf.placeholder(tf.int32, [None, 3], name='node_byxs')
        adj = tf.sparse_placeholder(tf.float32, [None, None], name='adj')
        node_feats = tf.gather_nd(self.conv_feats, node_byxs, name='node_feats')
        node_labels = tf.cast(tf.reshape(tf.gather_nd(self.labels, node_byxs), [-1]), tf.float32, name='node_labels')
        node_feats_resh = tf.expand_dims(node_feats, axis=0)

        gnn_feat_dropout = tf.placeholder_with_default(0., shape=())
        gnn_att_dropout = tf.placeholder_with_default(0., shape=())
        
        layer_name_list = []
        attns = []
        for head_idx in range(self.params.gat_n_heads[0]):
            cur_name = 'gat_hidden_1_%d'%(head_idx+1)
            layer_name_list.append(cur_name)
            attns.append(self.sp_attn_head(node_feats_resh, self.params.gat_hid_units[0], adj,
                                           name=cur_name,
                                           feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                           residual=self.params.gat_use_residual))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(self.params.gat_hid_units)):
            attns = []
            for head_idx in range(self.params.gat_n_heads[i]):
                cur_name = 'gat_hidden_%d_%d'%(i+1,head_idx+1)
                layer_name_list.append(cur_name)
                attns.append(self.sp_attn_head(h_1, self.params.gat_hid_units[i], adj,
                                               name=cur_name,
                                               feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                               residual=self.params.gat_use_residual))
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for head_idx in range(self.params.gat_n_heads[-1]):
            cur_name = 'gat_node_logits_%d'%(head_idx+1)
            layer_name_list.append(cur_name)
            out.append(self.sp_attn_head(h_1, 1, adj,
                                         name=cur_name,
                                         act=lambda x: x,
                                         feat_dropout=gnn_feat_dropout, att_dropout=gnn_att_dropout,
                                         residual=False)) 
        
        node_logits = tf.add_n(out) / self.params.gat_n_heads[-1]
        node_logits = tf.squeeze(node_logits, [0,2])
        
        # Hang up
        self.node_logits = node_logits            # [num_nodes,]
        self.gnn_final_feats = tf.squeeze(h_1)    # [num_nodes, sum(heads_i * hid_i)]
        self.node_byxs = node_byxs
        self.node_feats = node_feats
        self.node_labels = node_labels
        self.adj = adj
        self.gnn_feat_dropout = gnn_feat_dropout
        self.gnn_att_dropout = gnn_att_dropout
        self.var_to_restore += layer_name_list
        print("GAT part built.")
        
    def build_infer_module(self):
        print("Building the inference module...")
        # 'post_cnn' prefix retained
        post_cnn_dropout = tf.placeholder_with_default(0., shape=())
        
        is_lr_flipped = tf.placeholder(tf.bool, [])
        is_ud_flipped = tf.placeholder(tf.bool, [])
        rot90_num = tf.placeholder_with_default(0., shape=())
        
        # Grid from image H,W and win_size (integer math)
        H = tf.cast(tf.shape(self.imgs)[1], tf.float32)
        W = tf.cast(tf.shape(self.imgs)[2], tf.float32)
        SRNS_STRIDE = tf.cast(self.win_size, tf.float32)

        y_len = tf.cast(tf.math.ceil(H / SRNS_STRIDE), dtype=tf.int32)
        x_len = tf.cast(tf.math.ceil(W / SRNS_STRIDE), dtype=tf.int32)
        
        sp_size = tf.cond(tf.logical_or(tf.equal(rot90_num,0), tf.equal(rot90_num,2)),
                          lambda: tf.stack([y_len, x_len]),
                          lambda: tf.stack([x_len, y_len]))
            
        reshaped_gnn_feats = tf.reshape(tensor=self.gnn_final_feats,
                                        shape=tf.concat(values=[tf.slice(tf.shape(self.imgs),[0],[1]),
                                                                sp_size,
                                                                tf.slice(tf.shape(self.gnn_final_feats),[1],[1])], axis=0))
        
        reshaped_gnn_feats = tf.cond(is_lr_flipped, lambda: tf.image.flip_left_right(reshaped_gnn_feats), lambda: reshaped_gnn_feats)
        reshaped_gnn_feats = tf.cond(is_ud_flipped, lambda: tf.image.flip_up_down(reshaped_gnn_feats), lambda: reshaped_gnn_feats)
        reshaped_gnn_feats = tf.cond(tf.math.not_equal(rot90_num,0),
                                     lambda: tf.image.rot90(reshaped_gnn_feats, tf.cast(rot90_num, tf.int32)),
                                     lambda: reshaped_gnn_feats)
                                                    
        temp_num_chs = self.params.gat_n_heads[-2]*self.params.gat_hid_units[-1]

        post_cnn_conv_comp = self.new_conv_layer(reshaped_gnn_feats, [1,1,temp_num_chs,32], norm_type=self.params.norm_type, use_relu=True, name='post_cnn_conv_comp')
        current_input = post_cnn_conv_comp
        # Start deconv from win_size//2; snap to nearest available cnn_feat scale (powers of 2).
        ds_rate = int(self.win_size) // 2
        avail_ds = sorted(self.cnn_feat_spatial_sizes.keys())  # e.g., [1,2,4,8]
        if ds_rate not in avail_ds:
            # pick closest available starting scale to avoid shape mismatches (e.g., win_size=10 -> 5 -> start at 4)
            ds_rate = min(avail_ds, key=lambda k: abs(k - ds_rate))

        def _pick_ds(key):
            if key in self.cnn_feat_spatial_sizes:
                return key
            le = [k for k in avail_ds if k <= key]
            ge = [k for k in avail_ds if k >= key]
            if le:
                return le[-1]
            return ge[0] if ge else avail_ds[0]
        while ds_rate >= 1:
            cur_deconv_name = 'post_cnn_deconv%d'%(ds_rate)
            target_key = _pick_ds(ds_rate)

            # Dynamically compute target spatial size to avoid mismatches on non-power-of-two inputs (e.g., HRF).
            if ds_rate == 1:
                target_hw = tf.slice(tf.shape(self.imgs), [1], [2])  # match image spatial dims
            else:
                # Match the CNN feature spatial size we're going to concatenate with
                target_hw = tf.shape(self.cnn_feat[target_key])[1:3]

            # Resize + conv instead of transposed conv to avoid shape mismatches on HRF
            upsampled = tf.image.resize(current_input, target_hw, method=tf.image.ResizeMethod.BILINEAR)
            upsampled = tf.layers.conv2d(upsampled, filters=16, kernel_size=3, padding='same',
                                         activation=tf.nn.relu, name=cur_deconv_name)
            
            cur_cnn_feat = tf.nn.dropout(self.cnn_feat[target_key], 1-post_cnn_dropout)
            # Ensure spatial match before concat; resize if needed (HRF/non-power-of-two cases).
            cur_shape = tf.shape(cur_cnn_feat)[1:3]
            need_resize = tf.logical_or(tf.not_equal(cur_shape[0], target_hw[0]),
                                        tf.not_equal(cur_shape[1], target_hw[1]))
            cur_cnn_feat = tf.cond(need_resize,
                                   lambda: tf.image.resize(cur_cnn_feat, target_hw, method=tf.image.ResizeMethod.BILINEAR),
                                   lambda: cur_cnn_feat)
            if self.use_enc_layer: 
                cur_cnn_feat = self.new_conv_layer(cur_cnn_feat, [1,1,16,16], norm_type=self.params.norm_type, use_relu=True, name='post_cnn_cnn_feat%d'%(ds_rate))
            else:
                if self.params.norm_type=='GN':
                    num_grp = min(cfg.GN_MIN_NUM_G, max(1, 16 // max(1, cfg.GN_MIN_CHS_PER_G)))
                    cur_cnn_feat = self.group_norm_layer(cur_cnn_feat, num_group=num_grp, name='post_cnn_cnn_feat%d'%(ds_rate))
                    cur_cnn_feat = tf.nn.relu(cur_cnn_feat)

            if ds_rate == 1:
                cur_conv_name = 'post_cnn_img_output'
                output = self.new_conv_layer(tf.concat(values=[upsampled,cur_cnn_feat], axis=3),
                                             [self.infer_module_kernel_size,self.infer_module_kernel_size,32,1],
                                             name=cur_conv_name)
                self.post_cnn_img_output = output
            else:
                cur_conv_name = 'post_cnn_conv%d'%(ds_rate)
                output = self.new_conv_layer(tf.concat(values=[upsampled,cur_cnn_feat], axis=3),
                                             [self.infer_module_kernel_size,self.infer_module_kernel_size,32,32],
                                             norm_type=self.params.norm_type, use_relu=True, name=cur_conv_name)
            
            current_input = output
            ds_rate //= 2

        post_cnn_img_fg_prob = tf.sigmoid(current_input)
        
        pixel_weights = tf.placeholder(tf.float32, [None, None, None, 1], name='pixel_weights')
        
        # Hang up     
        self.post_cnn_dropout = post_cnn_dropout
        self.pixel_weights = pixel_weights
        self.post_cnn_img_fg_prob = post_cnn_img_fg_prob
        self.is_lr_flipped = is_lr_flipped
        self.is_ud_flipped = is_ud_flipped
        self.rot90_num = rot90_num
        self.reshaped_gnn_feats = reshaped_gnn_feats
        
        print("inference module built.")
        
    def build_optimizer(self):
        print("Building the optimizer part...")
        # helper to keep scheduler boundary dtype aligned with global_step
        def _piecewise_lr(values):
            boundary = int(self.params.max_iters * self.params.lr_decay_tp)
            boundary = max(0, boundary)
            boundary_dtype = self.global_step.dtype.base_dtype.as_numpy_dtype
            boundaries = [boundary_dtype(boundary)]
            return tf.train.piecewise_constant(self.global_step, boundaries, values)
        
        # ----- cnn related -----
        binary_mask_fg = tf.cast(tf.equal(self.labels, 1), tf.float32)
        binary_mask_bg = tf.cast(tf.not_equal(self.labels, 1), tf.float32)
        combined_mask = tf.concat(values=[binary_mask_bg,binary_mask_fg], axis=3)
        flat_one_hot_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
        flat_labels = tf.reshape(tensor=self.labels, shape=(-1,))
        flat_logits = tf.reshape(tensor=self.img_output, shape=(-1,))
        cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=tf.cast(flat_labels, tf.float32))
        
        num_pixel = tf.reduce_sum(self.fov_masks)
        num_pixel_fg = tf.math.count_nonzero(binary_mask_fg, dtype=tf.int64)
        num_pixel_bg = num_pixel - num_pixel_fg
        denom_pix = tf.cast(num_pixel, tf.float32) + cfg.EPSILON
        class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(tf.cast(num_pixel_fg, tf.float32),denom_pix),(1,1)),
                                                 tf.reshape(tf.divide(tf.cast(num_pixel_bg, tf.float32),denom_pix),(1,1))], axis=1), dtype=tf.float32)
        weight_per_label = tf.transpose(tf.matmul(flat_one_hot_labels, tf.transpose(class_weight)))
        reshaped_fov_masks = tf.reshape(tensor=tf.cast(self.fov_masks, tf.float32), shape=(-1,))
        reshaped_fov_masks /= (tf.reduce_mean(reshaped_fov_masks) + cfg.EPSILON)
        cnn_loss = tf.reduce_mean(reshaped_fov_masks * weight_per_label * cross_entropies)
        
        flat_bin_output = tf.greater_equal(tf.reshape(tensor=self.img_fg_prob, shape=(-1,)), 0.5)
        cnn_correct = tf.cast(tf.equal(flat_bin_output, tf.cast(flat_labels, tf.bool)), tf.float32)
        cnn_accuracy = tf.reduce_mean(cnn_correct)
        num_fg_output = tf.reduce_sum(tf.cast(flat_bin_output, tf.float32)) 
        cnn_tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), flat_bin_output), tf.float32))
        cnn_pre = tf.divide(cnn_tp, tf.add(num_fg_output,cfg.EPSILON))
        cnn_rec = tf.divide(cnn_tp, tf.cast(num_pixel_fg, tf.float32))

        # ----- gnn related -----
        gnn_cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.node_logits, labels=self.node_labels)
        num_node = tf.size(self.node_labels)
        num_node_fg = tf.count_nonzero(self.node_labels, dtype=tf.int32)
        num_node_bg = num_node - num_node_fg
        denom_node = tf.cast(num_node, tf.float32) + cfg.EPSILON
        gnn_class_weight = tf.cast(tf.concat(values=[tf.reshape(tf.divide(tf.cast(num_node_fg, tf.float32),denom_node),(1,1)),
                                                     tf.reshape(tf.divide(tf.cast(num_node_bg, tf.float32),denom_node),(1,1))], axis=1), dtype=tf.float32)
        gnn_weight_per_label = tf.transpose(tf.matmul(tf.one_hot(tf.cast(self.node_labels, tf.int32), 2), tf.transpose(gnn_class_weight)))
        gnn_loss = tf.cond(tf.greater(num_node, 0),
                           lambda: tf.reduce_mean(gnn_weight_per_label * gnn_cross_entropies),
                           lambda: tf.constant(0.0, tf.float32))

        gnn_prob = tf.sigmoid(self.node_logits)
        gnn_correct = tf.equal(tf.cast(tf.greater_equal(gnn_prob, 0.5), tf.int32), tf.cast(self.node_labels, tf.int32))
        gnn_accuracy = tf.reduce_mean(tf.cast(gnn_correct, tf.float32))
        
        # ----- inference module related -----
        post_cnn_flat_logits = tf.reshape(tensor=self.post_cnn_img_output, shape=(-1,))
        post_cnn_cross_entropies = tf.nn.sigmoid_cross_entropy_with_logits(logits=post_cnn_flat_logits, labels=tf.cast(flat_labels, tf.float32))
        reshaped_pixel_weights = tf.reshape(tensor=self.pixel_weights, shape=(-1,))
        reshaped_pixel_weights /= (tf.reduce_mean(reshaped_pixel_weights) + cfg.EPSILON)
        post_cnn_loss = tf.reduce_mean(reshaped_pixel_weights * weight_per_label * post_cnn_cross_entropies)
        
        post_cnn_flat_bin_output = tf.greater_equal(tf.reshape(tensor=self.post_cnn_img_fg_prob, shape=(-1,)), 0.5)
        post_cnn_correct = tf.cast(tf.equal(post_cnn_flat_bin_output, tf.cast(flat_labels, tf.bool)), tf.float32)
        post_cnn_accuracy = tf.reduce_mean(post_cnn_correct)
        post_cnn_num_fg_output = tf.reduce_sum(tf.cast(post_cnn_flat_bin_output, tf.float32))
        post_cnn_tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.cast(flat_labels, dtype=tf.bool), post_cnn_flat_bin_output), tf.float32))
        post_cnn_pre = tf.divide(post_cnn_tp, tf.add(post_cnn_num_fg_output,cfg.EPSILON))
        post_cnn_rec = tf.divide(post_cnn_tp, tf.cast(num_pixel_fg, tf.float32))

        # ----- joint optimization -----
        loss = tf.add_n([cnn_loss, post_cnn_loss]) if self.cnn_loss_on else post_cnn_loss
        weights_only = filter(lambda x: x.name.endswith('W:0'), tf.trainable_variables())
        weight_decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(x) for x in weights_only])) * cfg.TRAIN.WEIGHT_DECAY_RATE
        loss += weight_decay
        
        default_lr = float(getattr(self.params, 'lr', 1e-3))
        learning_rate = tf.placeholder_with_default(tf.constant(default_lr, dtype=tf.float32), shape=[], name='lr')
        self.lr_ph = learning_rate
        if self.params.opt=='adam':
            optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=0.1)
        elif self.params.opt=='sgd':
            optimizer = tf.train.MomentumOptimizer(learning_rate, cfg.TRAIN.MOMENTUM, use_nesterov=True)

        gvs_1 = optimizer.compute_gradients(loss)
        if self.gnn_loss_on:
            gvs_2 = optimizer.compute_gradients(gnn_loss*self.gnn_loss_weight)
            gvs_2 = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) if (('gat' in gv[1].name) and (gv[0] is not None)) else (None,gv[1]), gvs_2)
            gvs = map(lambda t: (add_tensors_wo_none([t[0][0], t[1][0]]), t[0][1]), list(zip(gvs_1, gvs_2)))
        else:
            gvs = gvs_1
            
        if self.params.old_net_ft_lr==0:
            # update only the newly added sub-network
            if self.params.lr_scheduling=='pc':
                values = [self.params.new_net_lr, self.params.new_net_lr*0.1]
                lr_handler = _piecewise_lr(values)
            elif self.params.lr_scheduling=='fixed':
                lr_handler = tf.constant(self.params.new_net_lr, dtype=tf.float32)
            elif self.params.lr_scheduling=='exp':
                decay_steps = max(1, int(self.params.max_iters/20))
                lr_handler = tf.train.exponential_decay(self.params.new_net_lr, self.global_step, decay_steps, 0.9, staircase=False)
            else:
                lr_handler = tf.constant(self.params.new_net_lr, dtype=tf.float32)
                
            if self.params.do_simul_training:
                gvs = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) if (('gat' in gv[1].name or 'post_cnn' in gv[1].name) and (gv[0] is not None)) else (None,gv[1]), gvs)
            else:
                gvs = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) if (('post_cnn' in gv[1].name) and (gv[0] is not None)) else (None,gv[1]), gvs)
                
            gvs = map(lambda gv: (gv[0]*self.params.infer_module_grad_weight,gv[1]) if 'post_cnn' in gv[1].name else (gv[0],gv[1]), gvs)
            train_op = optimizer.apply_gradients(gvs, global_step=self.global_step)
        else:
            # update the whole network
            if self.params.lr_scheduling=='pc':
                values = [self.params.old_net_ft_lr, self.params.old_net_ft_lr*0.1]
                lr_handler = _piecewise_lr(values)
            elif self.params.lr_scheduling=='fixed':
                lr_handler = tf.constant(self.params.old_net_ft_lr, dtype=tf.float32)
            elif self.params.lr_scheduling=='exp':
                decay_steps = max(1, int(self.params.max_iters/20))
                lr_handler = tf.train.exponential_decay(self.params.old_net_ft_lr, self.global_step, decay_steps, 0.9, staircase=False)
            else:
                lr_handler = tf.constant(self.params.old_net_ft_lr, dtype=tf.float32)

            lr_ratio = self.params.new_net_lr/self.params.old_net_ft_lr
            if self.params.do_simul_training:
                gvs = map(lambda gv: (lr_ratio*gv[0],gv[1]) if (('gat' in gv[1].name or 'post_cnn' in gv[1].name) and (gv[0] is not None)) else (gv[0],gv[1]), gvs)
            else:
                gvs = map(lambda gv: (lr_ratio*gv[0],gv[1]) if (('post_cnn' in gv[1].name) and (gv[0] is not None)) else (gv[0],gv[1]), gvs)
            gvs = map(lambda gv: (tf.clip_by_value(gv[0], -5., 5.),gv[1]) if gv[0] is not None else (gv[0],gv[1]), gvs)
            gvs = map(lambda gv: (gv[0]*self.params.infer_module_grad_weight,gv[1]) if 'post_cnn' in gv[1].name else (gv[0],gv[1]), gvs)
            train_op = optimizer.apply_gradients(gvs, global_step=self.global_step)

        # Hang up
        self.cnn_flat_bin_output = flat_bin_output
        self.cnn_loss = cnn_loss
        self.cnn_tp = cnn_tp
        self.cnn_num_fg_output = num_fg_output
        self.cnn_num_pixel_fg = num_pixel_fg
        self.cnn_accuracy = cnn_accuracy
        self.cnn_precision = cnn_pre        
        self.cnn_recall = cnn_rec
        
        self.gnn_prob = gnn_prob
        self.gnn_loss = gnn_loss
        self.gnn_accuracy = gnn_accuracy
        
        self.post_cnn_flat_bin_output = post_cnn_flat_bin_output
        self.post_cnn_loss = post_cnn_loss
        self.post_cnn_tp = post_cnn_tp
        self.post_cnn_num_fg_output = post_cnn_num_fg_output
        self.post_cnn_accuracy = post_cnn_accuracy
        self.post_cnn_precision = post_cnn_pre        
        self.post_cnn_recall = post_cnn_rec

        self.learning_rate = learning_rate
        self.lr_handler = lr_handler

        self.loss = loss
        self.train_op = train_op

        print("optimizer part built.")

############################################################
# Additional VGN functional builder (lightweight)
############################################################

def build_vgn():
    """Lightweight VGN graph for quick experiments."""
    images_ph = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='vgn_images')
    labels_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='vgn_labels')
    adj_ph = tf.placeholder(tf.float32, shape=[None, None], name='vgn_adjacency')
    node_pos_ph = tf.placeholder(tf.int32, shape=[None, 2], name='vgn_node_positions')
    is_training_ph = tf.placeholder(tf.bool, name='vgn_is_training')

    cnn_feats, cnn_prob = _vgn_build_cnn(images_ph, is_training_ph)

    with tf.name_scope('vgn_node_gather'):
        node_features = tf.gather_nd(
            cnn_feats,
            tf.concat([tf.zeros([tf.shape(node_pos_ph)[0], 1], dtype=tf.int32), node_pos_ph], axis=1)
        )

    gnn_output = _vgn_build_gnn(node_features, adj_ph, is_training_ph)
    fused_gnn_map = _vgn_scatter_to_image(gnn_output, node_pos_ph, tf.shape(cnn_prob), name='vgn_gnn_map')
    combined = tf.concat([cnn_prob, fused_gnn_map], axis=-1, name='vgn_combined')

    with tf.variable_scope('vgn_infer_head'):
        final_prob = tf.layers.conv2d(combined, filters=1, kernel_size=cfg.VGN.INFER_KERNEL_SIZE,
                                      padding='same', activation=tf.nn.sigmoid, name='vgn_final_prob')

    output_op = final_prob
    eps = 1e-8
    logits = tf.log(final_prob + eps) - tf.log(1. - final_prob + eps)
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_ph, logits=logits), name='vgn_loss')
    train_op = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.LR).minimize(loss_op, name='vgn_train_op')

    return images_ph, labels_ph, adj_ph, node_pos_ph, is_training_ph, loss_op, train_op, output_op

def _vgn_build_cnn(images, is_training):
    with tf.variable_scope('vgn_cnn'):
        images_norm = images - _vgn_get_pixel_mean()
        c1 = tf.layers.conv2d(images_norm, 64, 3, activation=tf.nn.relu, padding='same', name='conv1_1')
        c1 = tf.layers.conv2d(c1, 64, 3, activation=tf.nn.relu, padding='same', name='conv1_2')
        p1 = tf.layers.max_pooling2d(c1, 2, 2, name='pool1')
        c2 = tf.layers.conv2d(p1, 128, 3, activation=tf.nn.relu, padding='same', name='conv2_1')
        c2 = tf.layers.conv2d(c2, 128, 3, activation=tf.nn.relu, padding='same', name='conv2_2')
        p2 = tf.layers.max_pooling2d(c2, 2, 2, name='pool2')
        c3 = tf.layers.conv2d(p2, 256, 3, activation=tf.nn.relu, padding='same', name='conv3_1')
        c3 = tf.layers.conv2d(c3, 256, 3, activation=tf.nn.relu, padding='same', name='conv3_2')
        c3 = tf.layers.conv2d(c3, 256, 3, activation=tf.nn.relu, padding='same', name='conv3_3')
        feat_map = tf.layers.dropout(c3, rate=0.1, training=is_training, name='feat_dropout')
        prob_map = tf.layers.conv2d(feat_map, 1, 1, activation=tf.nn.sigmoid, name='cnn_prob')
    return feat_map, prob_map

def _vgn_build_gnn(node_features, adjacency, is_training):
    with tf.variable_scope('vgn_gnn'):
        C_in = node_features.get_shape().as_list()[-1]
        C_out = 16
        W = tf.get_variable('gat_W', shape=[C_in, C_out], initializer=_get_glorot_uniform_initializer())
        Wh = tf.matmul(node_features, W)
        a = tf.get_variable('gat_a', shape=[2 * C_out, 1], initializer=_get_glorot_uniform_initializer())
        N = tf.shape(node_features)[0]
        Wh_i = tf.tile(tf.expand_dims(Wh, 0), [N, 1, 1])
        Wh_j = tf.tile(tf.expand_dims(Wh, 1), [1, N, 1])
        attn_input = tf.concat([Wh_i, Wh_j], axis=-1)  # [N, N, 2*C_out]
        e = tf.tensordot(attn_input, a, axes=1)       # [N, N, 1]
        e = tf.nn.leaky_relu(tf.squeeze(e, -1), alpha=0.2)
        mask = (1. - adjacency) * 1e9
        e = e - mask
        alpha = tf.nn.softmax(e, axis=1)
        h_prime = tf.matmul(alpha, Wh)
        h_prime = tf.nn.relu(h_prime)
    return h_prime

def _vgn_scatter_to_image(node_values, node_positions, ref_shape, name=None):
    with tf.name_scope(name or 'vgn_scatter'):
        B = ref_shape[0]; H = ref_shape[1]; W = ref_shape[2]
        Fv = tf.shape(node_values)[1]
        out = tf.zeros([B, H, W, Fv], dtype=node_values.dtype)
        def body(i, acc):
            pos = node_positions[i]
            val = node_values[i]
            r = pos[0]; c = pos[1]
            one_hot = tf.reshape(val, [1,1,1,Fv])
            paddings = [[0,0],[r, H - r - 1],[c, W - c - 1],[0,0]]
            update = tf.pad(one_hot, paddings)
            return i + 1, acc + update
        i0 = tf.constant(0)
        N = tf.shape(node_positions)[0]
        _, scattered = tf.while_loop(lambda i, *_: i < N, body, [i0, out], parallel_iterations=32, back_prop=False)
        return scattered

def _vgn_get_pixel_mean():
    return tf.constant(cfg.PIXEL_MEAN_DRIVE, dtype=tf.float32)
