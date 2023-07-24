from Utils.utils import Option


def getHyper_bolT():

    hyperDict = {
        "batchSize": 16,
        "weightDecay": 0,
        "nOfEpochs": 20,
        "lr": 1e-4,

        # FOR BOLT
        "nOfLayers": 4,
        "dim": 400,
        "numHeads": 40,
        "headDim": 20,
        "windowSize": 20,
        "shiftCoeff": 2.0 / 5.0,
        "fringeCoeff":
        2,  # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
        "fringeRule": "expand",
        "weightMax": 1.0,
        "weightMin": 1.0,
        "mlpRatio": 1.0,
        "attentionBias": True,
        "qkvBias": False,
        "attnHeadScale": False,
        "ffGlu": False,
        "drop": 0.1,
        "attnDrop": 0.1,
        "lambdaCons": 0.1,

        # extra for ablation study
        "pooling": "cls",  # ["cls", "gmp"]
    }

    return Option(hyperDict)
