from Analysis.FID.fMRIToPng import normalizefMRITimeSeries, fMRIToPILImage, fcToPILImage, concatImageArraysVertically, concatImageArraysHorizontally
from Dataset.dataset import SupervisedDataset

from Utils.utils import getRecentRunPath, loadModel_distill_expert, writeParamsLog
from glob import glob
from Utils.fc import corrcoef
from Analysis.FID.fMRIToPng import fcToPILImage

from .hyperParams import hyperDict_dreamr

import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

import numpy as np


def sampleCounter(roiSignals, models, classifier, targetClass,
                  counterMethod):
    """
        models : [model_expert_0_T', ..., model_expert_T-T'_T]
        roiSignals : (B, N, T) in device
    """

    targetHyperDict = hyperDict_dreamr

    with torch.no_grad():
        originalLogits = classifier.getLogits(roiSignals)
        originalClassifiedLabels = originalLogits.argmax(dim=1)
        originalProbs = F.softmax(originalLogits, dim=-1)

    batchSize = roiSignals.shape[0]
    device = roiSignals.device

    counterRoiSignals_ = roiSignals.permute(0, 2, 1)
    noisyRoiSignals = None

    guidanceScale = targetHyperDict.baseGuidanceScale

    def guidedDenoise(counterRoiSignals, guidanceScale):

        T_start = int(targetHyperDict.timesteps *
                      targetHyperDict.iterationSchedule[k])

        print("Denosing from T_start = {}".format(T_start))

        time = torch.full((batchSize, ),
                          T_start,
                          device=device,
                          dtype=torch.long)

        log_snr = models[0].get_log_snr(time, roiSignals.shape)

        startModelIndex = int(
            np.ceil(T_start / targetHyperDict.timesteps *
                    targetHyperDict.expertCount))

        targetExpertIndexes = [
            startModelIndex - i - 1 for i in range(startModelIndex)
        ]

        print("targetExpertIndexes = {}".format(targetExpertIndexes))

        counterRoiSignals = counterRoiSignals.detach()
        counterRoiSignals = models[0].q_sample(counterRoiSignals, log_snr)

        noisyRoiSignals = counterRoiSignals.clone().cpu().detach()

        for i, expertIndex in enumerate(targetExpertIndexes):

            print("Denoising with expertIndex = {}".format(expertIndex))

            model = models[expertIndex]
            if (i == 0):
                T_start_input = T_start
            else:
                T_start_input = model.timeInterval[1]

            for m in model.modules():
                if isinstance(m, nn.Upsample):
                    m.recompute_scale_factor = None

            if (expertIndex == 0):

                counterRoiSignals, _, counterLabels, counterProbs = model.genCounterfact(
                    [], counterRoiSignals, classifier, targetClass, T_start_input, guidanceScale)

            else:

                counterRoiSignals, _ = model.genCounterfact(
                    models[:expertIndex], counterRoiSignals,
                    classifier, targetClass,
                    T_start_input, guidanceScale)

        return counterRoiSignals, noisyRoiSignals, counterLabels, counterProbs

    guidedCounterRoiSignals, noisyRoiSignals, counterLabels, counterProbs = guidedDenoise(counterRoiSignals_,
                                                                                          guidanceScale)

    counterRoiSignals = guidedCounterRoiSignals

    return counterRoiSignals.to("cpu").permute(
        0, 2, 1), noisyRoiSignals.permute(
            0, 2, 1), counterLabels.to("cpu"), originalClassifiedLabels.to(
                "cpu"), counterProbs.to("cpu"), originalProbs.to("cpu")


def gen_counter_dreamr(details):

    foldCount = details.foldCount  # ignore this
    targetDataset = details.targetDataset
    device = details.device
    classifierPath = details.classifierPath
    datePrepend = details.datePrepend
    nOfClasses = details.nOfClasses

    targetHyperDict = hyperDict_dreamr

    targetDistillIndex = targetHyperDict.genTargetDistillIndex
    distilledTimesteps = int(targetHyperDict.timesteps /
                             np.power(2, targetDistillIndex))

    if (details.targetRunFolder != None):
        recentRunPath = details.targetRunFolder
    else:
        recentRunPath = getRecentRunPath(
            "dreamr", targetDataset)  # retrieve the recent train

    print(
        "Generating counter samples for targetdataset = {} using recentRunPath = {}"
        .format(targetDataset, recentRunPath))

    dataset = SupervisedDataset(targetDataset, details.dynamicLength,
                                targetHyperDict.batchSize, foldCount)

    for fold in range(foldCount):

        # TODO for fast research at the moment, delete for full results
        if (fold > 0):
            break

        foldPath = recentRunPath + "/fold{}/DISTILL_{}".format(
            fold, distilledTimesteps)

        modelPath = foldPath + "/ModelSaves/"

        classifierPath_fold = classifierPath + "/model_fold{}.save".format(
            fold)

        print("classifierPath_fold = {}".format(classifierPath_fold))

        classifier = torch.load(classifierPath_fold, map_location=device)

        genPath = foldPath + "/CounterSamples/gen-{}".format(datePrepend)
        os.makedirs(genPath, exist_ok=True)

        writeParamsLog(genPath + "/hyperDict_{}.txt".format("dreamr"),
                       targetHyperDict.dict)

        models = []

        for expertIndex in range(targetHyperDict.expertCount):

            epoch = -1
            print("foldPath : {}, distilledTimesteps = {}, expertIndex : {}".
                  format(foldPath, distilledTimesteps, expertIndex))
            loaded = loadModel_distill_expert(modelPath, distilledTimesteps,
                                              expertIndex, device)
            model = loaded[0]
            model.device = device
            epoch = loaded[1]

            models.append(model)

        print("Loaded models for fold : {}, epoch : {}".format(fold, epoch))

        for targetClass in range(nOfClasses):

            counterBag = []

            counterPath = genPath + "/toClass_{}".format(targetClass)
            os.makedirs(counterPath, exist_ok=True)

            imageSavePath = counterPath + "/fid_samples"
            os.makedirs(imageSavePath, exist_ok=True)

            fcImageSavePath = counterPath + "/fid_samples_fc"
            os.makedirs(fcImageSavePath, exist_ok=True)

            visualInspectionPath = counterPath + "/inspection"
            os.makedirs(visualInspectionPath, exist_ok=True)

            print("Generating counter samples for test with targetClass = {}".
                  format(targetClass))

            dataset.setFold_gen(fold, targetClass, train=False)
            lengthGroups = dataset.lengthGroups

            for lengthGroup in lengthGroups:

                dataLoader = dataset.getSet_gen(lengthGroup,
                                                targetHyperDict.batchSize_gen)

                for data in tqdm(dataLoader, ncols=60, ascii=True):

                    xTests = data["timeseries"]
                    yTests = data["label"]
                    subjIds = data["subjId"]

                    if (xTests.shape[2] % 2 != 0):
                        xTests = xTests[:, :, :-1]

                    if (xTests.shape[2] == 274):
                        xTests = xTests[:, :, :-2]

                    if (xTests.shape[2] == 290):
                        xTests = xTests[:, :, :-2]

                    counterSamples, noisySamples, counterLabels, originalClassifiedLabels, counterProbs, originalProbs = sampleCounter(
                        details.methodName, xTests.to(device), models,
                        classifier, targetClass,
                        targetHyperDict.counterMethod)

                    for i, counterSample in enumerate(counterSamples):

                        success = counterLabels[i] == targetClass

                        counterBag.append({
                            "isTest":
                            True,
                            "success":
                            success,
                            "subjId":
                            subjIds[i],
                            "targetClass":
                            targetClass,
                            "originalLabel":
                            yTests[i],
                            'prob_before_for_targetClass_{}'.format(targetClass):
                            originalProbs[i].detach().cpu().numpy(),
                            "originalClassifiedLabel":
                            originalClassifiedLabels[i],
                            'prob_after_for_targetClass_{}'.format(targetClass):
                            counterProbs[i].detach().cpu().numpy(),
                            "counterLabel":
                            counterLabels[i],
                            "timeseries":
                            xTests[i].numpy(),
                            "counter_timeseries":
                            counterSample.numpy(),
                            "noisy_timeseries":
                            noisySamples[i].numpy()
                        })

                        print("counterSample.shape = {}, xTests[i].shape = {}".
                              format(counterSample.shape, xTests[i].shape))

                        resultImage = concatImageArraysHorizontally([
                            xTests[i], counterSample, noisySamples[i],
                            counterSample - xTests[i]
                        ])

                        fc = corrcoef(xTests[i])
                        fcCounter = corrcoef(counterSample)

                        fcResultImage = concatImageArraysHorizontally(
                            [fc, fcCounter, fcCounter - fc])

                        fcResultImage = fcToPILImage(fcResultImage)
                        fcResultImage.save(
                            visualInspectionPath +
                            "/subjId_{}_oldLabel_{}_newLabel_{}_fc.png".format(
                                subjIds[i], originalClassifiedLabels[i],
                                counterLabels[i]))

                        resultImage = fMRIToPILImage(resultImage)

                        resultImage.save(
                            visualInspectionPath +
                            "/subjId_{}_oldLabel_{}_newLabel_{}.png".format(
                                subjIds[i], originalClassifiedLabels[i],
                                counterLabels[i]))

                        counterSample = normalizefMRITimeSeries(
                            counterSample.numpy())
                        image = fMRIToPILImage(counterSample)
                        image.save(
                            imageSavePath +
                            "/subjId_{}_oldLabel_{}_newLabel_{}.png".format(
                                subjIds[i], originalClassifiedLabels[i],
                                counterLabels[i]))

                        fcImage = fcToPILImage(fcCounter)
                        fcImage.save(
                            fcImageSavePath +
                            "/subjId_{}_oldLabel_{}_newLabel_{}.png".format(
                                subjIds[i], originalClassifiedLabels[i],
                                counterLabels[i]))

                # for train, to see whatsup

                dataset.setFold_gen(fold, targetClass, train=True)
                lengthGroups = dataset.lengthGroups

                for lengthGroup in lengthGroups:

                    dataLoader = dataset.getSet_gen(
                        lengthGroup, targetHyperDict.batchSize_gen)

                    print(
                        "Generating counter samples for train with targetClass = {}"
                        .format(targetClass))
                    for data in tqdm(dataLoader, ncols=60, ascii=True):

                        xTrains = data["timeseries"]
                        yTrains = data["label"]
                        subjIds = data["subjId"]

                        if (xTrains.shape[2] % 2 != 0):
                            xTrains = xTrains[:, :, :-1]

                        if (xTrains.shape[2] == 274):
                            xTrains = xTrains[:, :, :-2]

                        if (xTrains.shape[2] == 290):
                            xTrains = xTrains[:, :, :-2]

                        counterSamples, noisySamples, counterLabels, originalClassifiedLabels, counterProbs, originalProbs, _ = sampleCounter(
                            details.methodName, xTrains.to(device), models,
                            classifier, targetClass,
                            targetHyperDict.counterMethod)

                        for i, counterSample in enumerate(counterSamples):

                            success = counterLabels[i] == targetClass

                            counterBag.append({
                                "isTest":
                                False,
                                "success":
                                success,
                                "subjId":
                                subjIds[i],
                                "targetClass":
                                targetClass,
                                "originalLabel":
                                yTrains[i],
                                'prob_before_for_targetClass_{}'.format(targetClass):
                                originalProbs[i].detach().cpu().numpy(),
                                "originalClassifiedLabel":
                                originalClassifiedLabels[i],
                                'prob_after_for_targetClass_{}'.format(targetClass):
                                counterProbs[i].detach().cpu().numpy(),
                                "counterLabel":
                                counterLabels[i],
                                "timeseries":
                                xTrains[i].numpy(),
                                "counter_timeseries":
                                counterSample.numpy(),
                                "noisy_timeseries":
                                noisySamples[i].numpy()
                            })

            torch.save(counterBag, counterPath + "/counterBag.save")
