import torch
from torch import nn

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from .util import windowBoldSignal

import math

# # only for 69
# class GELU(torch.nn.Module):

#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return torch.nn.functional.gelu(input)

# torch.nn.modules.activation.GELU = GELU


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale, nW=None):
    """
        x : (b * nW, t, c)
        shift : (b, c)
        scale : (b, c)

        if nW is None
            x : (b, t, c)
            shift : (b, c)
            scale : (b, c)
    """

    if (nW != None):
        x = rearrange(x, "(b nW) t c -> b nW t c", nW=nW)
        shift = shift.unsqueeze(1).unsqueeze(2).repeat(1, nW, x.shape[2], 1)
        scale = scale.unsqueeze(1).unsqueeze(2).repeat(1, nW, x.shape[2], 1)
    else:
        shift = shift.unsqueeze(1).repeat(1, x.shape[1], 1)
        scale = scale.unsqueeze(1).repeat(1, x.shape[1], 1)

    # print("x.shape = ", x.shape)
    # print("shift.shape = ", shift.shape)
    # print("scale.shape = ", scale.shape)

    x = x * (1 + scale) + shift

    if (nW != None):
        x = rearrange(x, "b nW t c -> (b nW) t c")

    return x


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(
        self,
        dim,
        mult=1,
        dropout=0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        activation = nn.GELU()

        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):

    def __init__(self,
                 dim,
                 windowSize,
                 receptiveSize,
                 numHeads,
                 headDim=20,
                 attentionBias=True,
                 qkvBias=True,
                 attnDrop=0.,
                 projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize  # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim**-0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias

        maxDisparity = windowSize - 1 + (receptiveSize - windowSize) // 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * maxDisparity + 1, numHeads))  # maxDisparity, nH

        self.cls_bias_sequence_up = nn.Parameter(
            torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(
            torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # get pair-wise relative position index for each token inside the window
        coords_x = torch.arange(self.windowSize)  # N
        coords_x_ = torch.arange(self.receptiveSize) - (
            self.receptiveSize - self.windowSize) // 2  # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # N, M
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)

        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.cls_bias_sequence_up, std=.02)
        trunc_normal_(self.cls_bias_sequence_down, std=.02)
        trunc_normal_(self.cls_bias_self, std=.02)

        self.softmax = nn.Softmax(dim=-1)

        # for token painting
        # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionMaps = None
        # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None
        self.nW = None

    def forward(self, x, x_, mask, nW, padMask):
        """
            Input:

            x: base BOLD tokens with shape of (B*num_windows, 1+windowSize, C), the first one is cls token
            x_: receptive BOLD tokens with shape of (B*num_windows, 1+receptiveSize, C), again the first one is cls token
            mask: (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize)
            nW: number of windows
            padMask : None or (padEffectedWindowCount, 1+windowSize, 1+windowReceptiveSize )

            Output:

            transX : attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the first one is cls token

        """

        B_, N, C = x.shape
        _, M, _ = x_.shape
        N = N - 1
        M = M - 1

        B = B_ // nW

        mask_left, mask_right = mask

        # linear mapping
        q = self.q(x)  # (batchSize * #windows, 1+N, C)
        k, v = self.kv(x_).chunk(2, dim=-1)  # (batchSize * #windows, 1+M, C)

        # head seperation
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        attn = torch.matmul(q, k.transpose(
            -1, -2)) * self.scale  # (batchSize*#windows, h, n, m)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, N, M

        if (self.attentionBias):
            attn[:, :, 1:,
                 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :,
                 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down

        # mask the not matching queries and tokens here
        maskCount = mask_left.shape[0]
        # repate masks for batch and heads
        mask_left = repeat(mask_left,
                           "nM nn mm -> b nM h nn mm",
                           b=B,
                           h=self.numHeads)
        mask_right = repeat(mask_right,
                            "nM nn mm -> b nM h nn mm",
                            b=B,
                            h=self.numHeads)

        mask_value = max_neg_value(attn)

        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW=nW)

        # make sure masks do not overflow
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        attn[:, :maskCount].masked_fill_(mask_left == 1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right == 1, mask_value)

        # now mask if there is 0 pad for the last window
        if (padMask != None):
            padEffectedWindowCount = padMask.shape[0]
            padMask = repeat(padMask,
                             "pW nn mm -> b pW h nn mm",
                             b=B,
                             h=self.numHeads)
            attn[:, -padEffectedWindowCount:].masked_fill_(
                padMask == 1, mask_value)

        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")

        attn = self.softmax(attn)  # (b, h, n, m)

        attn = self.attnDrop(attn)

        x = torch.matmul(attn, v)  # of shape (b_, h, n, d)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)
        x = self.projDrop(x)

        return x


class FusedWindowTransformer(nn.Module):

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads,
                 headDim, mlpRatio, attentionBias, drop, attnDrop):

        super().__init__()

        self.attention = WindowAttention(dim=dim,
                                         windowSize=windowSize,
                                         receptiveSize=receptiveSize,
                                         numHeads=numHeads,
                                         headDim=headDim,
                                         attentionBias=attentionBias,
                                         attnDrop=attnDrop,
                                         projDrop=drop)

        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, timeEmbeddings, cls, windowX, windowX_, mask, nW,
                padMask, padCount):
        """

            Input: 

            x : (B, T, C)
            timeEmbeddings : (B, C)
            cls : (B, nW, C)
            windowX: (B*nW, 1+windowSize, C)
            windowX_ (B*nW, 1+windowReceptiveSize, C)
            mask: (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize)
            nW : number of windows
            padMask : None or (padEffectedWindowCount, 1+windowSize, 1+windowReceptiveSize )
            padCount : number of zeros added to original x

            Output:

            xTrans : (B, T, C)
            clsTrans : (B, nW, C)

        """

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            timeEmbeddings).chunk(6, dim=1)

        # WINDOW ATTENTION
        windowXTrans = self.attention(
            modulate(self.attn_norm(windowX), shift_msa, scale_msa, nW),
            modulate(self.attn_norm(windowX_), shift_msa, scale_msa, nW), mask,
            nW, padMask)  # (B*nW, 1+windowSize, C)
        clsTrans = windowXTrans[:, :1]  # (B*nW, 1, C)
        xTrans = windowXTrans[:, 1:]  # (B*nW, windowSize, C)

        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)
        # FUSION
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)

        # residual connections
        clsTrans = clsTrans * gate_msa.unsqueeze(dim=1).repeat(
            1, clsTrans.shape[1], 1) + cls
        xTrans = xTrans * gate_msa.unsqueeze(dim=1).repeat(
            1, xTrans.shape[1], 1) + x

        # MLP layers
        xTrans = xTrans + self.mlp(
            modulate(self.mlp_norm(xTrans), shift_mlp, scale_mlp)
        ) * gate_mlp.unsqueeze(dim=1).repeat(1, xTrans.shape[1], 1)
        clsTrans = clsTrans + self.mlp(
            modulate(self.mlp_norm(clsTrans), shift_mlp, scale_mlp)
        ) * gate_mlp.unsqueeze(dim=1).repeat(1, clsTrans.shape[1], 1)

        # enforce padded ones to be zero
        if (padCount != 0):
            xTrans[:, -padCount:] = 0

        return xTrans, clsTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        """
        Input:
            windowedX : (batchSize, nW, windowLength, C)
            scatterWeights : (windowLength, )

        Output:
            destination: (batchSize, dynamicLength, C)

        """

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]

        device = windowedX.device

        destination = torch.zeros((batchSize, dynamicLength, C)).to(device)
        scalerDestination = torch.zeros(
            (batchSize, dynamicLength, C)).to(device)

        indexes = torch.tensor(
            [[j + (i * shiftSize) for j in range(windowLength)]
             for i in range(nW)]).to(device)
        indexes = indexes[None, :, :, None].repeat(
            (batchSize, 1, 1, C))  # (batchSize, nW, windowSize, featureDim)

        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")

        destination.scatter_add_(dim=1, index=indexes, src=src)

        scalerSrc = torch.ones(
            (windowLength)).to(device)[None, None, :, None].repeat(
                batchSize, nW, 1,
                C)  # (batchSize, nW, windowLength, featureDim)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")

        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)

        destination = destination / scalerDestination

        return destination


class BolTransformerBlock(nn.Module):

    def __init__(self,
                 dim,
                 numHeads,
                 headDim,
                 windowSize,
                 receptiveSize,
                 shiftSize,
                 mlpRatio=1.0,
                 drop=0.0,
                 attnDrop=0.0,
                 attentionBias=True):

        assert ((receptiveSize - windowSize) % 2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim,
                                                  windowSize=windowSize,
                                                  shiftSize=shiftSize,
                                                  receptiveSize=receptiveSize,
                                                  numHeads=numHeads,
                                                  headDim=headDim,
                                                  mlpRatio=mlpRatio,
                                                  attentionBias=attentionBias,
                                                  drop=drop,
                                                  attnDrop=attnDrop)

        self.timeEmbedder = TimestepEmbedder(dim)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        self.L = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        maskCount = self.L // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize + 1,
                                self.receptiveSize + 1)
        mask_right = torch.zeros(maskCount, self.windowSize + 1,
                                 self.receptiveSize + 1)

        for i in range(maskCount):
            if (self.L > 0):
                mask_left[i, :, 1:1 + self.L - shiftSize * i] = 1
                mask_right[maskCount - 1 - i, :, -self.L + shiftSize * i:] = 1

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)

    def forward(self, x, t, cls):
        """
        Input:
            x : (batchSize, T, C)
            t : (batchSize, )
            cls : (batchSize, nW, C)
            time_embeddings : (batchSize, C)

        Output:
            fusedX_trans : (batchSize, T, C)
            cls_trans : (batchSize, nW, C)

        """

        B, T, C = x.shape
        device = x.device

        # if x falls short in sequence length, then pad with zeros
        Z = self.windowSize + self.shiftSize * (cls.shape[1] - 1)

        P = Z - T

        if (P > 0):
            x = torch.cat([x, torch.zeros((B, P, C), device=device)], dim=1)
            effectedWindowCount = (P + self.L) // self.shiftSize + 1

            padMask = torch.zeros(effectedWindowCount,
                                  self.windowSize + 1,
                                  self.receptiveSize + 1,
                                  device=device)
            for i in range(effectedWindowCount):
                padMask[-i, :, -self.L - P + i * self.shiftSize:-self.L +
                        i * self.shiftSize] = 1
        else:
            padMask = None

        # form the padded x to be used for focal keys and values
        x_ = torch.cat([
            torch.zeros((B, self.L, C), device=device), x,
            torch.zeros((B, self.L, C), device=device)
        ],
            dim=1)  # (B, remainder+Z+remainder, C)

        timeEmbeddings = self.timeEmbedder(t)  # (B, C)

        # window the sequences
        windowedX, _ = windowBoldSignal(x.transpose(
            2, 1), self.windowSize, self.shiftSize)  # (B, nW, C, windowSize)
        windowedX = windowedX.transpose(2, 3)  # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(
            x_.transpose(2, 1), self.receptiveSize,
            self.shiftSize)  # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2, 3)  # (B, nW, receptiveSize, C)

        nW = windowedX.shape[1]  # number of windows

        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX],
                         dim=2)  # (B, nW, 1+windowSize, C)
        xcls = rearrange(xcls,
                         "b nw l c -> (b nw) l c")  # (B*nW, 1+windowSize, C)

        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_],
                          dim=2)  # (B, nw, 1+receptiveSize, C)
        xcls_ = rearrange(
            xcls_, "b nw l c -> (b nw) l c")  # (B*nW, 1+receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        # pass to fused window transformer
        fusedX_trans, cls_trans = self.transformer(
            x, timeEmbeddings, cls, xcls, xcls_, masks, nW, padMask,
            P)  # (B*nW, 1+windowSize, C)

        fusedX_trans = fusedX_trans[:, :T]

        return fusedX_trans, cls_trans
