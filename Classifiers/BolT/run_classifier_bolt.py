from audioop import cross
from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys
import copy

from scipy.stats import mode

from Utils.variables import resultsSaveDir

from datetime import datetime

from Utils.utils import Option, writeDictToFile, writeParamsLog
from Utils.metrics import metricSummer_oneSeed, calculateMetric, calculateMetrics

from .code.model import Model
from Dataset.dataset import SupervisedDataset

from .hyperParams import getHyper_bolT


def train(model, dataLoader, nOfEpochs):

    print("\nTraining started\n")

    for epoch in range(nOfEpochs):

        preds = []
        probs = []
        groundTruths = []
        losses = []

        for i, data in enumerate(tqdm(dataLoader, ncols=60, ascii=True)):

            xTrain = data["timeseries"]  # (batchSize, N, dynamicLength)
            yTrain = data["label"]  # (batchSize, )

            # NOTE: xTrain and yTrain are still on "cpu" at this point

            train_loss, train_preds, train_probs, yTrain = model.step(
                xTrain, yTrain, train=True)

            torch.cuda.empty_cache()

            preds.append(train_preds)
            probs.append(train_probs)
            groundTruths.append(yTrain)
            losses.append(train_loss)

        preds = torch.cat(preds, dim=0).numpy()
        probs = torch.cat(probs, dim=0).numpy()
        groundTruths = torch.cat(groundTruths, dim=0).numpy()
        losses = torch.tensor(losses).numpy()

        metrics = calculateMetric({
            "predictions": preds,
            "probs": probs,
            "labels": groundTruths
        })
        print("Epoch : {} Train metrics : {}".format(epoch, metrics))

        # model.schedulerStep()

    return preds, probs, groundTruths, losses


def test(model, dataLoader, isVal=False):

    preds = []
    probs = []
    groundTruths = []
    losses = []

    for i, data in enumerate(dataLoader):

        xTest = data["timeseries"]
        yTest = data["label"]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        test_loss, test_preds, test_probs, yTest = model.step(xTest,
                                                              yTest,
                                                              train=False)

        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()

    metrics = calculateMetric({
        "predictions": preds,
        "probs": probs,
        "labels": groundTruths
    })
    if (isVal):
        print("\n \n Val metrics : {}".format(metrics))
    else:
        print("\n \n Test metrics : {}".format(metrics))

    return preds, probs, groundTruths, loss


def run_bolT(details):

    savePath = resultsSaveDir + "/Classifiers/BolT/{}".format(
        details.targetDataset)

    os.makedirs(savePath, exist_ok=True)

    hyperParams = getHyper_bolT()

    writeParamsLog(savePath + "/hyperDict.txt", hyperParams.dict)

    # extract datasetDetails

    batchSize = hyperParams.batchSize
    nOfEpochs = hyperParams.nOfEpochs

    foldCount = details.foldCount
    dynamicLength = details.dynamicLength

    dataset = SupervisedDataset(details.targetDataset, dynamicLength,
                                batchSize, foldCount)

    details = Option({
        "device": details.device,
        "nOfTrains": None,
        "nOfClasses": dataset.nOfClasses,
        "batchSize": batchSize,
        "nOfEpochs": nOfEpochs,
        "inputDim": dataset.inputDim
    })

    results = []

    for fold in range(foldCount):

        # no optimization

        dataLoader_train = dataset.getFold(fold=fold, train=True)

        details.nOfTrains = len(dataLoader_train)
        model = Model(hyperParams, details)

        train_preds, train_probs, train_groundTruths, train_loss = train(
            model, dataLoader_train, nOfEpochs)

        dataLoader_test = dataset.getFold(fold=fold, train=False)
        test_preds, test_probs, test_groundTruths, test_loss = test(
            model, dataLoader_test)

        result = {
            "train": {
                "labels": train_groundTruths,
                "predictions": train_preds,
                "probs": train_probs,
                "loss": train_loss
            },
            "test": {
                "labels": test_groundTruths,
                "predictions": test_preds,
                "probs": test_probs,
                "loss": test_loss
            }
        }

        results.append({"result_outerFold": result})

        outerFold_result_metric = {
            "train": calculateMetric(result["train"]),
            "test": calculateMetric(result["test"])
        }

        fileName_results = savePath + "/results.txt"
        file_results = open(fileName_results, "a")
        file_results.write("\n\n Fold : {} \n\n".format(
            fold, outerFold_result_metric))
        file_results.write("Train = {} \n\n".format(
            outerFold_result_metric["train"]))
        file_results.write("Test = {}\n".format(
            outerFold_result_metric["test"]))
        file_results.write("\n\n")
        file_results.close()

        torch.save(model, savePath + "/model_fold{}.save".format(fold))

        break  # only the first fold is needed at the development stage bro letsss go

    torch.save(results, savePath + "/results.save")

    print("\n \n resultss = {}".format(results[0].keys()))
    metrics = calculateMetrics(results)

    print("\n \n metrics = {}".format(metrics))
    meanMetric_all, stdMetric_all = metricSummer_oneSeed(metrics, "test")

    # now dump metrics

    resultFileName = "results_mean.txt"
    metrics_file = open(savePath + "/{}".format(resultFileName), "w")
    metrics_file.write("\n -- TEST MEAN -- \n")
    writeDictToFile(meanMetric_all, metrics_file, 0)
    metrics_file.write("\n -- TEST STD -- \n")
    writeDictToFile(stdMetric_all, metrics_file, 0)
    metrics_file.close()

    print("mean test metric : {}".format(meanMetric_all))
    print("std test metric : {}".format(stdMetric_all))
