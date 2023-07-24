from Utils.utils import Option


def getHyper_bolTDiffusion_adaIN():

    hyperDict = {
        "lr": 2e-4,

        # FOR BOLT
        "nOfLayers": 4,  # use to be 6
        "dim": 800,
        "numHeads": 40,
        "headDim": 20,
        "windowSize": 50,
        "shiftCoeff": 4.0 / 5.0,
        "fringeCoeff":
        2,  # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
        "fringeRule": "shrink",
        "weightMax": 1.0,
        "weightMin": 1.0,
        "mlpRatio": 1.0,
        "attentionBias": True,
        "qkvBias": False,
        "attnHeadScale": False,
        "ffGlu": False,
        "drop": 0.0,
        "attnDrop": 0.0,
    }

    return Option(hyperDict)
