from Utils.utils import Option

hyperDict_dreamr = Option({
    "batchSize": 8,  # 8
    "saveSamplesEvery": 5,
    "nOfEpochs": 200,  # 100,
    "nOfEpochs_distill": 10,  # 20,
    ####
    "noiseSchedule": "cosine",  # "linear" or "sigma", "cosine"
    "expertCount": 4,
    "lossType": "l1",
    "timesteps": 1024,
    "finalTimesteps": 4,
    "objective": "pred_v",  # "pred_noise", "pred_v"
    "emaDecay": 0.0,
    # for gen
    "sampleLength":
    256,  # for the length of the generated samples during training
    "samplingTimesteps": 8,
    "baseGuidanceScale":
    160.0,
    "ddim_eta": 0.0,
    "genTargetDistillIndex": 7,
    "batchSize_gen": 1,
})
