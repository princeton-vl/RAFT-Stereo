import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler

try:
    import corr_sampler
except ImportError:
    corr_sampler = None

class CorrSampler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, coords, radius):
        if corr_sampler is None:
            raise RuntimeError(
                "reg_cuda requires the optional sampler extension; build sampler/ "
                "or use reg_gather for the build-free PyTorch path"
            )
        ctx.save_for_backward(volume,coords)
        ctx.radius = radius
        corr, = corr_sampler.forward(volume, coords, radius)
        return corr
    @staticmethod
    def backward(ctx, grad_output):
        volume, coords = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_volume, = corr_sampler.backward(volume, coords, grad_output, ctx.radius)
        return grad_volume, None, None

class CorrBlockFast1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        # all pairs correlation
        corr = CorrBlockFast1D.corr(fmap1, fmap2)
        batch, h1, w1, dim, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, 1, w2)
        for i in range(self.num_levels):
            self.corr_pyramid.append(corr.view(batch, h1, w1, -1, w2//2**i))
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])

    def __call__(self, coords):
        out_pyramid = []
        bz, _, ht, wd = coords.shape
        coords = coords[:, :1]  # list indexing ([0]) builds a CPU index tensor -> H2D copy + sync
        for i in range(self.num_levels):
            corr = CorrSampler.apply(self.corr_pyramid[i].squeeze(3), (coords/2**i).float(), self.radius)  # sampler reads fp32 coords
            out_pyramid.append(corr.view(bz, -1, ht, wd))
        return torch.cat(out_pyramid, dim=1)

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class PytorchAlternateCorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):
        B, D, H, W = fmap2.shape
        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        output_corr = []
        for grid_slice in grid.unbind(3):
            fmapw_mini = F.grid_sample(fmap2, grid_slice, align_corners=True)
            corr = torch.sum(fmapw_mini * fmap1, dim=1)
            output_corr.append(corr)
        corr = torch.stack(output_corr, dim=1).permute(0,2,3,1)

        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape
        fmap1 = self.fmap1
        fmap2 = self.fmap2
        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.zeros(1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)
            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2).clone()
            centroid_lvl[...,0] = centroid_lvl[...,0] / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)
            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, [1, 2], stride=[1, 2])
            out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).float()  # caller picks the memory format


class CorrBlock1D:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock1D.corr(fmap1, fmap2)

        batch, h1, w1, _, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, 1, 1, w2)
        # lookup offsets, created on the device once (a CPU tensor .to(device) per call syncs the stream)
        self.dx = torch.linspace(-radius, radius, 2*radius+1, device=corr.device).view(2*radius+1, 1)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels):
            corr = F.avg_pool2d(corr, [1,2], stride=[1,2])
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords[:, :1].permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            x0 = self.dx + coords.reshape(batch*h1*w1, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            coords_lvl = torch.cat([x0,y0], dim=-1)
            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).float()  # caller picks the memory format

    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr / torch.sqrt(torch.tensor(D).float())


class CorrBlockGather1D:
    """Same lookup as reg / reg_cuda, in plain PyTorch ops that export to ONNX / TensorRT (no grid_sample, no extension):
    all-pairs volume as one batched matmul in the feature dtype (fp16 under AMP, as reg_cuda), pyramid by 1-D average
    pooling, linear interpolation of the 2r+2 integer taps around each lookup position via torch.gather (ONNX
    GatherElements). Taps outside the row read 0 (= reg_cuda / grid_sample zero padding). Output channels are
    level-major (level * (2r+1) + tap), as reg_cuda."""

    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        B, D, H, W1 = fmap1.shape
        # [B, H, W1, D] @ [B, H, D, W2] -> [B, H, W1, W2]
        corr = torch.matmul(fmap1.permute(0, 2, 3, 1), fmap2.permute(0, 2, 1, 3)) * (D**-0.5)
        corr = corr.reshape(B * H * W1, -1)
        self.shape = (B, H, W1)
        self.corr_pyramid = [corr]
        for _ in range(1, num_levels):
            # == F.avg_pool1d(corr, 2, 2) as the mean of even / odd slices (TensorRT 11's Myelin finds no tactic for
            # a 1-D average pool over B*H*W rows)
            n = corr.shape[-1] // 2
            corr = corr[:, : 2 * n].reshape(corr.shape[0], n, 2).mean(dim=-1)
            self.corr_pyramid.append(corr)
        self.taps = torch.arange(-radius, radius + 2, device=fmap1.device)

    def __call__(self, coords):
        B, H, W1 = self.shape
        x = coords[:, 0].reshape(B * H * W1, 1).float()
        out = []
        for i, vol in enumerate(self.corr_pyramid):
            w2 = vol.shape[-1]
            xi = x / 2**i
            x0 = torch.floor(xi)
            a = (xi - x0).to(vol.dtype)
            idx = x0.long() + self.taps  # [N, 2r+2]
            valid = (idx >= 0) & (idx < w2)
            v = torch.gather(vol, 1, idx.clamp(0, w2 - 1)) * valid.to(vol.dtype)
            out.append(v[:, :-1] * (1 - a) + v[:, 1:] * a)  # [N, 2r+1]
        out = torch.cat(out, dim=1).view(B, H, W1, -1)
        return out.permute(0, 3, 1, 2)  # NHWC storage (channels_last view)
