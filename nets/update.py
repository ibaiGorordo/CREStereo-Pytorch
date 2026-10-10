import torch
import torch.nn as nn
import torch.nn.functional as F

#Ref: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(BasicMotionEncoder, self).__init__()
        
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        # under autocast `out` is fp16 and flow fp32: the cat would promote every GRU input to fp32 only for the next
        # conv to round it back; rounding flow here gives the same conv inputs
        return torch.cat([out, flow.to(out.dtype)], dim=1)


class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(BasicUpdateBlock, self).__init__()

        self.encoder = BasicMotionEncoder(cor_planes)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, mask_size**2 *9, 1, padding=0))

    @torch.no_grad()
    def prepare_inference(self):
        """Inference: merged z|r convs, and the GRU conv weights split by input part so that the constant context part
        (inp, the same in every iteration of a stage) is convolved once per stage (see context()). GRU conv input
        channels: [h (hidden) | inp (hidden) | motion features (128)]. conv(cat[a, b], W) = conv(a, W_a) + conv(b, W_b)
        (zero padding is linear), so the result is the same up to rounding."""
        g = self.gru
        hd = g.convz1.weight.shape[0]
        for d in (1, 2):
            wz, wr, wq = (getattr(g, f"conv{n}{d}").weight for n in "zrq")
            bz, br, bq = (getattr(g, f"conv{n}{d}").bias for n in "zrq")
            wzr = torch.cat([wz, wr], 0)
            for name, t in (
                (f"zr_h{d}", torch.cat([wzr[:, :hd], wzr[:, 2 * hd :]], 1)),
                (f"zr_i{d}", wzr[:, hd : 2 * hd]),
                (f"zr_b{d}", torch.cat([bz, br])),
                (f"q_h{d}", torch.cat([wq[:, :hd], wq[:, 2 * hd :]], 1)),
                (f"q_i{d}", wq[:, hd : 2 * hd]),
                (f"q_b{d}", bq),
            ):
                self.register_buffer("_" + name, t.contiguous(), persistent=False)
        self.fused = True
        self._prepared_version = self._weight_version()
        return self

    def _weight_version(self):
        params = tuple(self.gru.parameters())
        if any(p.is_inference() for p in params):
            return None
        return tuple((id(p), p._version, p.device, p.dtype) for p in params)

    def context(self, inp):
        """per-stage constant part of the four GRU convs (needs prepare_inference)."""
        if not (
            torch.jit.is_tracing() or torch.onnx.is_in_onnx_export() or torch.compiler.is_compiling()
        ):
            version = self._weight_version()
            if version is None:
                return inp
            if version != self._prepared_version:
                self.prepare_inference()
        return [
            (
                F.conv2d(inp, self._zr_i1, self._zr_b1, padding=(0, 2)),
                F.conv2d(inp, self._q_i1, self._q_b1, padding=(0, 2)),
            ),
            (
                F.conv2d(inp, self._zr_i2, self._zr_b2, padding=(2, 0)),
                F.conv2d(inp, self._q_i2, self._q_b2, padding=(2, 0)),
            ),
        ]

    def _gru_fused(self, h, ctx, mo):
        hd = h.shape[1]
        for d, pad in ((1, (0, 2)), (2, (2, 0))):
            czr, cq = ctx[d - 1]
            zr = torch.sigmoid(
                F.conv2d(torch.cat([h, mo], 1), getattr(self, f"_zr_h{d}"), None, padding=pad) + czr
            )
            z, r = zr[:, :hd], zr[:, hd:]
            q = torch.tanh(
                F.conv2d(torch.cat([r * h, mo], 1), getattr(self, f"_q_h{d}"), None, padding=pad) + cq
            )
            h = (1 - z) * h + z * q
        return h

    def forward(self, net, inp, corr, flow, upsample=True, compute_mask=True):
        # print(inp.shape, corr.shape, flow.shape)
        motion_features = self.encoder(flow, corr)
        if isinstance(inp, (list, tuple)):  # precomputed context(inp) (prepare_inference)
            net = self._gru_fused(net, inp, motion_features)
        else:
            inp = torch.cat((inp, motion_features), dim=1)
            net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        if not compute_mask:
            return net, None, delta_flow

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow
