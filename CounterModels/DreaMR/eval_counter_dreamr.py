from Dataset.dataset import SupervisedDataset
from Utils.utils import getRecentGenPath, getRecentRunPath, writeDictToFile
from Utils.counterfactMetrics import calculateFID, calculateFlipRate, calculateProximity, calculateSparsity, calculateMeanProbability, calculateWassersteinDistanceOfFCs, calculateMSEOfFCs
import numpy as np
import torch


def eval_counter_dreamr(details):

    foldCount = details.foldCount
    targetDataset = details.targetDataset
    device = details.device
    nOfClasses = details.nOfClasses
    isVal = details.isVal

    if (details.targetRunFolder != None):
        recentRunPath = details.targetRunFolder
    else:
        recentRunPath = getRecentRunPath(
            "dreamr", targetDataset)  # retrieve the recent train

    dataset = SupervisedDataset(targetDataset, None, 1, foldCount)

    for fold in range(foldCount):

        dataset.setFold(fold, train=False)

        validationSubjIds = dataset.validationSubjIds

        # TODO
        if (fold > 0):
            break

        if (details.targetGenFolders != None):
            recentGenPath = details.targetGenFolders[fold]
        else:
            recentGenPath = getRecentGenPath(recentRunPath, fold)

        foldPath = recentGenPath

        classResults = []

        for targetClass in range(nOfClasses):

            print("Evaluating for class {}...".format(targetClass))

            classPath = foldPath + "/toClass_{}".format(targetClass)

            counterBag = torch.load(classPath + "/counterBag.save")

            print("Calculating Wasserstein Distance of FCs...")
            wassertein_distance = calculateWassersteinDistanceOfFCs(
                counterBag, targetDataset, isVal, validationSubjIds)
            print("Wasserstein Distance: {}".format(wassertein_distance))

            print("Calculating Flip Rate...")
            flipRate = calculateFlipRate(counterBag, targetClass, isVal,
                                         validationSubjIds)
            print("Calculating Proximity...")
            proximity, proximity_fc = calculateProximity(
                counterBag, isVal, validationSubjIds)
            print("Proximity: {}".format(proximity))

            print("Calculating Sparsity...")
            sparsity, sparsity_fc = calculateSparsity(counterBag, isVal,
                                                      validationSubjIds)
            print("Sparsity: {}".format(sparsity))

            print("Calculating Mean Probability...")
            meanProbability = calculateMeanProbability(counterBag, targetClass,
                                                       isVal,
                                                       validationSubjIds)

            print("Calculating FID...")
            fid1, fid2 = calculateFID(classPath + "/fid_samples",
                                      targetDataset, device, isVal,
                                      validationSubjIds, False)

            fid1_fc, fid2_fc = calculateFID(
                classPath + "/fid_samples_fc",
                targetDataset,
                device,
                isVal,
                validationSubjIds,
                True,
            )

            mse = calculateMSEOfFCs(counterBag, targetDataset, isVal,
                                    validationSubjIds)

            resultDict = {
                "flipRate": flipRate,
                "proximity": proximity,
                "sparsity": sparsity,
                "fid1": fid1,
                "fid2": fid2,
                "mse": mse,
                "fid1_fc": fid1_fc,
                "fid2_fc": fid2_fc,
                "proximity_fc": proximity_fc,
                "sparsity_fc": sparsity_fc,
                "wassertein_distance": wassertein_distance,
                "meanProbability": meanProbability
            }

            classResults.append(resultDict)

            if (isVal):
                resultFile = open(classPath + "/results_val.txt", "w")
            else:
                resultFile = open(classPath + "/results_test.txt", "w")

            writeDictToFile(resultDict, resultFile, 0)
            resultFile.close()

            if (isVal):
                torch.save(resultDict, foldPath + "/results_val.save")
            else:
                torch.save(resultDict, foldPath + "/results_test.save")

        if (isVal):
            resultFile = open(foldPath + "/results_val.txt", "w")
        else:
            resultFile = open(foldPath + "/results_test.txt", "w")
        # average the results of class array
        averageResultMean = {}
        averageResultStd = {}
        for resultDict in classResults:
            for key in resultDict:
                if (key not in averageResultMean):
                    averageResultMean[key] = []
                    averageResultStd[key] = []
                averageResultMean[key].append(resultDict[key])
                averageResultStd[key].append(resultDict[key])

        for key in averageResultMean:
            averageResultMean[key] = np.mean(averageResultMean[key])
            averageResultStd[key] = np.std(averageResultStd[key])

        writeDictToFile(averageResultMean, resultFile, 0)
        writeDictToFile(averageResultStd, resultFile, 0)
        resultFile.close()
