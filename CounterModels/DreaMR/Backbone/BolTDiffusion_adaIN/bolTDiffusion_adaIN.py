import torch
from torch import nn

import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# import transformers

from .bolTransformerBlock import BolTransformerBlock


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BolTDiffusion_adaIN(nn.Module):

    def __init__(self, hyperParams, details):

        super().__init__()

        dim = hyperParams.dim
        headDim = hyperParams.headDim
        inputDim = details.inputDim

        if (hasattr(details, "selfCondition")):
            self.selfCondition = details.selfCondition
        else:
            self.selfCondition = False

        self.hyperParams = hyperParams

        self.inputNorm = nn.LayerNorm(dim)

        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = []

        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        self.shiftSize = shiftSize
        self.receptiveSizes = []

        #headDim = headDim * 2 if hyperParams.selfCondition else headDim

        for i, layer in enumerate(range(hyperParams.nOfLayers)):
            j = i

            if (hyperParams.fringeRule == "expand"):
                receptiveSize = hyperParams.windowSize + math.ceil(
                    hyperParams.windowSize * 2 * j * hyperParams.fringeCoeff *
                    (1 - hyperParams.shiftCoeff))
            elif (hyperParams.fringeRule == "fixed"):
                receptiveSize = hyperParams.windowSize + math.ceil(
                    hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff *
                    (1 - hyperParams.shiftCoeff))
            elif (hyperParams.fringeRule == "shrink"):
                receptiveSize = hyperParams.windowSize + math.ceil(
                    hyperParams.windowSize * 2 *
                    (hyperParams.nOfLayers - j - 1) * hyperParams.fringeCoeff *
                    (1 - hyperParams.shiftCoeff))

            receptiveSize = min(500, receptiveSize)
            receptiveSize = max(50, receptiveSize)

            if ((receptiveSize - hyperParams.windowSize) % 2 != 0):
                receptiveSize += 1

            print("receptiveSize per window for layer {} : {}".format(
                i, receptiveSize))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(
                BolTransformerBlock(dim=dim,
                                    numHeads=hyperParams.numHeads,
                                    headDim=headDim,
                                    windowSize=hyperParams.windowSize,
                                    receptiveSize=receptiveSize,
                                    shiftSize=shiftSize,
                                    mlpRatio=hyperParams.mlpRatio,
                                    attentionBias=hyperParams.attentionBias,
                                    drop=hyperParams.drop,
                                    attnDrop=hyperParams.attnDrop))

        self.blocks = nn.ModuleList(self.blocks)

        timeDim = dim * 4

        if (self.selfCondition):
            self.inputProjector = nn.Linear(inputDim * 2, dim)
        else:
            self.inputProjector = nn.Linear(inputDim, dim)

        self.outputProjector = nn.Linear(dim, inputDim)

        self.initializeWeights()

    def initializeWeights(self):
        # a bit arbitrary
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def forward(self, roiSignals, time=None, roiSignals_condition=None):
        """
            Input : 

                roiSignals : (batchSize, T, R)
                time : (batchSize, )
                boldTokens_condition : (batchSize, T, N)


            Output:

                outputBoldTokens : (batchSize, #layers, T, N)
                outputClsTokens : (batchSize, #layers, #windows, N)

        """

        batchSize = roiSignals.shape[0]
        T = roiSignals.shape[1]  # dynamicLength

        nW = math.ceil((T - self.hyperParams.windowSize) / self.shiftSize) + 1
        cls = self.clsToken.repeat(batchSize, nW,
                                   1)  # (batchSize, #windows, C)

        boldTokens = roiSignals

        if (self.selfCondition):
            roiSignals_condition = default(
                roiSignals_condition, lambda: torch.zeros_like(roiSignals))
            roiSignals = torch.cat([roiSignals, roiSignals_condition], dim=2)

        boldTokens = self.inputProjector(
            roiSignals)  # (batchSize, T, R) -> (batchSize, T, N)

        for block in self.blocks:
            boldTokens, cls = block(boldTokens, time, cls)

        boldTokens = self.outputProjector(boldTokens)
        """
            boldTokens : (batchSize, T, R)
            cls : (batchSize, nW, N)
        """

        return boldTokens, cls
