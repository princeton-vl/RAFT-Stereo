import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D, CorrBlockGather1D
from core.utils.utils import coords_grid, upflow8


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class RAFTStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # channels_last keeps every activation NHWC, which cuDNN's tensor-core kernels use natively
        # (NCHW inputs make cuDNN wrap convolutions in nchwToNhwc/nhwcToNchw transposes)
        self.memory_format = torch.channels_last if getattr(args, 'channels_last', False) else torch.contiguous_format
        # channels_last_loop: NHWC only in the update loop; the encoders stay NCHW (InstanceNorm on NHWC is slow:
        # it is computed as batch_norm on a reshaped tensor, which copies)
        self.loop_memory_format = torch.channels_last if getattr(args, 'channels_last_loop', False) else self.memory_format

        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    @torch.no_grad()
    def fold_batchnorm(self):
        """Inference: eval BatchNorm layers that directly follow a conv folded into that conv (in fp32; exact up to
        rounding). Removes the BN passes and keeps exported fp16 graphs single-typed (graph optimizers that fold an
        fp32 BN into an fp16 conv produce mixed-type convs TensorRT rejects)."""

        def fold(conv, bn):
            s = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
            b = conv.bias.float() if conv.bias is not None else torch.zeros_like(s)
            conv.weight.data = (conv.weight.float() * s.view(-1, 1, 1, 1)).to(conv.weight.dtype)
            conv.bias = nn.Parameter(
                ((b - bn.running_mean.float()) * s + bn.bias.float()).to(conv.weight.dtype)
            )

        for m in self.modules():
            for c, n in (("conv1", "norm1"), ("conv2", "norm2"), ("conv3", "norm3")):
                if isinstance(getattr(m, c, None), nn.Conv2d) and isinstance(
                    getattr(m, n, None), nn.BatchNorm2d
                ):
                    fold(getattr(m, c), getattr(m, n))
                    setattr(m, n, nn.Identity())
            ds = getattr(m, "downsample", None)
            if (
                isinstance(ds, nn.Sequential)
                and len(ds) == 2
                and isinstance(ds[0], nn.Conv2d)
                and isinstance(ds[1], nn.BatchNorm2d)
            ):
                fold(ds[0], ds[1])
                ds[1] = nn.Identity()
                if isinstance(getattr(m, "norm3", None), nn.BatchNorm2d):
                    m.norm3 = nn.Identity()
        return self

    def half_conv_weights(self):
        """Inference with mixed precision: store conv weights in fp16 once. Autocast would cast them
        fp32 -> fp16 again in every iteration (its cast cache is cleared whenever the autocast region
        is left); the rounding is the same, so the output is unchanged. BatchNorm is folded into the convs first."""
        self.fold_batchnorm()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.half()
        return self

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        # disparity state; fp16 only to study precision (fp16 spacing is 0.25 for values 256-512, i.e. 1 px at full res)
        dtype = getattr(torch, getattr(self.args, 'coords_dtype', 'float32'))
        coords0 = coords_grid(N, H, W, device=img.device).to(dtype)
        coords1 = coords_grid(N, H, W, device=img.device).to(dtype)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.reshape(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        # 3x3 neighbourhoods as 9 shifted slices of the zero-padded flow (= F.unfold(.., [3,3], padding=1), same
        # order; unfold/Im2Col has no ONNX / TensorRT counterpart)
        fp = F.pad(factor * flow, [1, 1, 1, 1])
        up_flow = torch.stack([fp[:, :, i:i + H, j:j + W] for i in range(3) for j in range(3)], dim=2)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, D, factor*H, factor*W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # feed the network in its weight dtype: a no-op for fp32 weights, the same rounding autocast applies for
        # fp16 weights, and required when the whole model is fp16 (model.half(), no autocast)
        w_dtype = self.update_block.encoder.convf1.weight.dtype
        image1 = (2 * (image1 / 255.0) - 1.0).to(w_dtype).contiguous(memory_format=self.memory_format)
        image2 = (2 * (image2 / 255.0) - 1.0).to(w_dtype).contiguous(memory_format=self.memory_format)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                if getattr(self.args, 'fnet_sequential', False):
                    # one image at a time: halves the encoder's full-res activation peak; InstanceNorm is
                    # per sample, so the result is the same as the batched call
                    fmap1, fmap2 = self.fnet(image1), self.fnet(image2)
                else:
                    fmap1, fmap2 = self.fnet([image1, image2])
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning 
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]
            if self.loop_memory_format != self.memory_format:
                net_list = [x.contiguous(memory_format=self.loop_memory_format) for x in net_list]
                inp_list = [[x.contiguous(memory_format=self.loop_memory_format) for x in l] for l in inp_list]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "reg_cuda":
            corr_block = CorrBlockFast1D
        elif self.args.corr_implementation == "reg_gather": # reg_cuda's lookup in plain PyTorch (exportable, no extension)
            corr_block = CorrBlockGather1D
        else:
            raise ValueError("supported correlation implementations: reg, alt, reg_cuda and reg_gather")
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1).contiguous(memory_format=self.loop_memory_format) # index correlation volume
            flow = (coords1 - coords0).to(w_dtype).contiguous(memory_format=self.loop_memory_format)  # NCHW flow would turn every cat it joins NCHW
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                # in test_mode only the last iteration's upsampling mask is used
                compute_mask = not test_mode or itr == iters-1
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2, compute_mask=compute_mask)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
