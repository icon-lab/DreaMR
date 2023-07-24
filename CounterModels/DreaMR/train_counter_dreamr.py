import enum
import numpy as np

from Analysis.FID.fMRIToPng import normalizefMRITimeSeries, fMRIToPILImage, concatImageArraysVertically

from Utils.utils import Option, getRecentRunPath, loadModel_distill_expert, saveModel_distill_expert, writeParamsLog, writeMessageLog

from einops import rearrange, repeat
from .model_dreamr import Model
from tqdm import tqdm
import torch
from Utils.variables import resultsSaveDir
import os
import math
from Dataset.dataset import SupervisedDataset

from torchvision import utils, transforms

from glob import glob

import time

from .hyperParams import hyperDict_dreamr


def sampleFromModels(models, sampleShape):

    randomRoiSignal = torch.randn(sampleShape)

    for i, model in enumerate(models):
        print("Sampling from expert : {} of {}".format(i, len(models)))
        randomRoiSignal = randomRoiSignal.to(model.device)
        randomRoiSignal = model.sample(randomRoiSignal)

    return randomRoiSignal


def sampleAndSave(models, sampleShape, batchSize, targetFolder, epoch):

    sample_times = sampleShape[0] // batchSize

    images = []

    transform = transforms.ToTensor()

    for i in range(sample_times):

        samples = sampleFromModels(models,
                                   [batchSize, sampleShape[1], sampleShape[2]])

        for sample in samples:
            sample = sample.detach().cpu().numpy()  # (T, R)

            sample = rearrange(sample, "T R -> R T")

            sample = normalizefMRITimeSeries(sample)
            image = fMRIToPILImage(sample)

            image = transform(image)

            images.append(image)

    utils.save_image(images,
                     targetFolder + "/epoch_{}.png".format(epoch),
                     nrow=int((sampleShape[0])**(0.5)))


def train_counter_dreamr(details):

    foldCount = details.foldCount
    targetDataset = details.targetDataset
    datePrepend = details.datePrepend
    fromExists = details.fromExists

    targetHyperDict = hyperDict_dreamr

    timesteps = targetHyperDict.timesteps
    finalTimesteps = targetHyperDict.finalTimesteps

    assert (timesteps % finalTimesteps == 0)

    distillationCount = int(np.log(timesteps // finalTimesteps) / np.log(2))

    saveSamplesEvery = targetHyperDict.saveSamplesEvery
    device = details.device

    methodName = "dreamr"

    savePath = resultsSaveDir + "/Counterfacts/{}/{}/run-{}".format(
        methodName, targetDataset, datePrepend)

    if (details.targetRunFolder != None):
        savePath = details.targetRunFolder
    else:
        if (fromExists):
            savePath = getRecentRunPath(
                methodName, targetDataset)  # retrieve the recent train

    os.makedirs(savePath, exist_ok=True)

    writeParamsLog(savePath + "/hyperDict_{}.txt".format(methodName),
                   targetHyperDict.dict)
    writeParamsLog(savePath + "/trainDetails.txt", details.dict)

    dataset = SupervisedDataset(targetDataset, details.dynamicLength,
                                targetHyperDict.batchSize, foldCount)

    epochCount_top = targetHyperDict.nOfEpochs
    epochCount_distill = targetHyperDict.nOfEpochs_distill

    models_student = None
    models_teacher = None

    for fold in range(foldCount):

        # TODO for fast research at the moment, delete for full results
        if (fold > 0):
            break

        for distillationIndex in range(distillationCount + 1):

            if (models_student != None):
                del models_student
                del models_teacher

            torch.cuda.empty_cache()

            distilledTimesteps = int(timesteps /
                                     (np.power(2, distillationIndex)))

            distillPath = savePath + "/fold{}/DISTILL_{}".format(
                fold, distilledTimesteps)
            os.makedirs(distillPath, exist_ok=True)

            modelPath = distillPath + "/ModelSaves/"
            os.makedirs(modelPath, exist_ok=True)

            logPath = distillPath + "/log".format(fold)
            os.makedirs(logPath, exist_ok=True)

            # now load teach model if needed

            models_teacher = []

            if (distillationIndex != 0):

                epochCount = epochCount_distill

                distilledTimesteps_teacher = int(
                    timesteps / (np.power(2, distillationIndex - 1)))
                distillPath_teacher = savePath + "/fold{}/DISTILL_{}/ModelSaves/".format(
                    fold, distilledTimesteps_teacher)

                for expertIndex in range(targetHyperDict.expertCount):

                    loaded = loadModel_distill_expert(
                        distillPath_teacher, distilledTimesteps_teacher,
                        expertIndex, device)
                    model_ = loaded[0]
                    model_.device = device

                    models_teacher.append(model_)

            else:

                epochCount = epochCount_top

            # now load student models

            models_student = []

            epoch = -1

            dataLoader = dataset.getFold(fold, train=True)

            for expertIndex in range(targetHyperDict.expertCount):

                details = Option({
                    "device": device,
                    "stepsPerEpoch": len(dataLoader),
                    "inputDim": dataset.inputDim,
                    "dynamicLength": details.dynamicLength,
                    "expertIndex": expertIndex,
                    "expertCount": targetHyperDict.expertCount,
                    "distillationIndex": distillationIndex,
                    "methodName": details.methodName,
                })

                epoch = -1

                print("target model path = {}".format(modelPath))
                loaded = loadModel_distill_expert(modelPath,
                                                  distilledTimesteps,
                                                  expertIndex, device)

                # print the number of elements in dataLoader

                if (loaded != None):
                    model_ = loaded[0]
                    model_.device = device
                    epoch = loaded[1] + 1
                else:
                    model_ = Model(targetHyperDict, details)

                    if (len(models_teacher) > 0):
                        model_.copy_from_teacher(models_teacher[expertIndex])
                        print("copying from teacher")

                models_student.append(model_)

            if (epoch >= epochCount):

                print("Distillation Index : {} has already finished, skipping".
                      format(distillationIndex))
                continue

            for i in range(epochCount):

                if (epoch == epochCount):
                    break

                if (epoch == -1):
                    epoch = 0

                outer_losses = []

                for expertIndex in range(targetHyperDict.expertCount):

                    model_student = models_student[expertIndex]

                    if (distillationIndex != 0):

                        model_teacher = models_teacher[expertIndex]

                        inner_losses = []

                        for data in tqdm(dataLoader, ncols=60, ascii=True):
                            xTrain = data["timeseries"]
                            loss = model_student.step(xTrain, device,
                                                      model_teacher)

                            inner_losses.append(loss)

                        message = "Epoch : {}, Distill : {}, Expert : {}, loss = {}"\
                            .format(epoch, distilledTimesteps, expertIndex, np.mean(inner_losses))

                        print(message)
                        writeMessageLog(logPath + "/log.txt", message)

                        if (epoch % saveSamplesEvery == 0 and epoch != 0):
                            print(
                                "Saving model for epoch : {}, expertIndex : {}"
                                .format(epoch, expertIndex))
                            saveModel_distill_expert(model_student, modelPath,
                                                     distilledTimesteps, epoch,
                                                     expertIndex)

                        del xTrain

                        torch.cuda.empty_cache()

                        if (epoch == epochCount - 1):
                            saveModel_distill_expert(model_student, modelPath,
                                                     distilledTimesteps, epoch,
                                                     expertIndex)
                    else:

                        inner_losses = []

                        for data in tqdm(dataLoader, ncols=60, ascii=True):

                            xTrain = data["timeseries"]

                            loss = model_student.step(xTrain, device)

                            inner_losses.append(loss)

                        message = "Epoch : {}, Expert : {}, loss = {}"\
                            .format(epoch, expertIndex, np.mean(inner_losses))

                        print(message)
                        writeMessageLog(logPath + "/log.txt", message)

                        if (epoch % saveSamplesEvery == 0 and epoch != 0):
                            print(
                                "Saving model for epoch : {}, expertIndex : {}"
                                .format(epoch, expertIndex))
                            saveModel_distill_expert(model_student, modelPath,
                                                     distilledTimesteps, epoch,
                                                     expertIndex)

                        del xTrain

                        torch.cuda.empty_cache()

                        if (epoch == epochCount - 1):
                            saveModel_distill_expert(model_student, modelPath,
                                                     distilledTimesteps, epoch,
                                                     expertIndex)

                    outer_losses.append(np.mean(inner_losses))

                if (epoch % saveSamplesEvery == 0 and epoch != 0):
                    print("Sampling for epoch : {}".format(epoch))
                    sampleAndSave(
                        list(reversed(models_student)),
                        (targetHyperDict.batchSize,
                         targetHyperDict.sampleLength, details.inputDim),
                        targetHyperDict.batchSize,
                        logPath,
                        epoch,
                        isDime=details.methodName == "dime")
                    print("Sampling done")

                message = "Epoch : {}, Average loss = {}"\
                    .format(epoch, np.mean(outer_losses))

                print(message)
                writeMessageLog(logPath + "/log.txt", message)

                epoch += 1
