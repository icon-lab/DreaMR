from Utils.fc import corrcoef
from Utils.fid.fid_score import compute_statistics_of_path_, calculate_frechet_distance
from Utils.utils import getRecentRunPath
import torch
import numpy as np
from Utils.utils import writeDictToFile
from Utils.utils import getRecentGenPath
from tqdm import tqdm
import ot


def mean_squared_error(matrix1, matrix2):
    n_samples1 = matrix1.shape[0]
    n_samples2 = matrix2.shape[0]

    # Initialize the MSE matrix
    mse_matrix = np.zeros((n_samples1, n_samples2))

    # Compute Mean Squared Error
    for i in range(n_samples1):
        for j in range(n_samples2):
            mse_matrix[i, j] = np.mean((matrix1[i] - matrix2[j])**2)

    return mse_matrix


def calculateMSEOfFCs(counterBag, targetDataset, isVal, validationSubjIds):

    targetFCs = []

    # print("validationSubjIds = ", validationSubjIds)
    print("isVal = ", isVal)

    numberOfValidation = 0
    numberOfTest = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        fc_counter = corrcoef(counter["counter_timeseries"])

        fc_counter = fc_counter[np.triu_indices_from(fc_counter, k=1)]

        # print("counter['subjId'] = ", counter["subjId"])

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                numberOfValidation += 1
                continue
            numberOfTest += 1

        targetFCs.append(fc_counter.flatten())

    targetFCs = np.array(targetFCs)

    # here we load the source FCs
    loaded = torch.load("path/to/fcArray.save")

    sourceFCs = loaded[0]
    sourceSubjIds = loaded[1]

    sourceFCs = sourceFCs.reshape(sourceFCs.shape[0], -1)

    batchSize = 8

    if (targetDataset == "hcpTask_0"):
        batchSize = 1

    nOfSourceSamples = sourceFCs.shape[0]
    nOfTargetSamples = targetFCs.shape[0]

    runningSimilarity = 0
    j = 0

    for i in tqdm(range(0, nOfTargetSamples, batchSize), ncols=60, ascii=True):

        left = i
        right = min(i + batchSize, nOfTargetSamples)

        j += (right - left)

        similarity = mean_squared_error(targetFCs[left:right], sourceFCs)

        runningSimilarity += np.mean(np.max(similarity,
                                            axis=1)) * (right - left)

    runningSimilarity /= j

    return runningSimilarity


def pearson_correlation(matrix1, matrix2):
    n_samples1 = matrix1.shape[0]
    n_samples2 = matrix2.shape[0]

    # Initialize the correlation matrix
    correlation_matrix = np.zeros((n_samples1, n_samples2))

    # Center the matrices
    centered_matrix1 = matrix1 - np.mean(matrix1, axis=1, keepdims=True)
    centered_matrix2 = matrix2 - np.mean(matrix2, axis=1, keepdims=True)

    # Compute Pearson Correlation
    numerator = np.dot(centered_matrix1, centered_matrix2.T)
    denominator = np.sqrt(np.sum(centered_matrix1 ** 2, axis=1, keepdims=True)) * \
        np.sqrt(np.sum(centered_matrix2 ** 2, axis=1, keepdims=True)).T

    correlation_matrix = numerator / denominator

    return correlation_matrix


def calculatePearsonCorrelationOfFCs(counterBag, targetDataset, isVal,
                                     validationSubjIds):

    targetFCs = []

    # print("validationSubjIds = ", validationSubjIds)
    print("isVal = ", isVal)

    numberOfValidation = 0
    numberOfTest = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        fc_counter = corrcoef(counter["counter_timeseries"])

        fc_counter = fc_counter[np.triu_indices_from(fc_counter, k=1)]

        # print("counter['subjId'] = ", counter["subjId"])

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                numberOfValidation += 1
                continue
            numberOfTest += 1

        targetFCs.append(fc_counter.flatten())

    targetFCs = np.array(targetFCs)

    if (targetDataset == "hcpRest_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpRest/hcpRest_fcArray.save"
        )
    elif (targetDataset == "hcpTask_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpTask/hcpTask_fcArray.save"
        )
    elif (targetDataset == "id1000_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/id1000/id1000_fcArray.save"
        )

    sourceFCs = loaded[0]
    sourceSubjIds = loaded[1]

    sourceFCs = sourceFCs.reshape(sourceFCs.shape[0], -1)

    targetFCs = sourceFCs

    batchSize = 8

    if (targetDataset == "hcpTask_0"):
        batchSize = 1

    nOfSourceSamples = sourceFCs.shape[0]
    nOfTargetSamples = targetFCs.shape[0]

    runningSimilarity = 0
    j = 0

    for i in tqdm(range(0, nOfTargetSamples, batchSize), ncols=60, ascii=True):

        left = i
        right = min(i + batchSize, nOfTargetSamples)

        j += (right - left)

        similarity = pearson_correlation(targetFCs[left:right], sourceFCs)

        runningSimilarity += np.mean(similarity) * (right - left)

    runningSimilarity /= j

    return runningSimilarity


def cosine_similarity(matrix1, matrix2):
    # Normalize rows in matrix1
    matrix1_norm = np.linalg.norm(matrix1, axis=1, keepdims=True)
    matrix1_normalized = matrix1 / matrix1_norm

    # Normalize rows in matrix2
    matrix2_norm = np.linalg.norm(matrix2, axis=1, keepdims=True)
    matrix2_normalized = matrix2 / matrix2_norm

    # Compute cosine similarity
    similarity = np.dot(matrix1_normalized, matrix2_normalized.T)

    return similarity


def calculateCosineDistanceOfFCs(counterBag, targetDataset, isVal,
                                 validationSubjIds):

    targetFCs = []

    # print("validationSubjIds = ", validationSubjIds)
    print("isVal = ", isVal)

    numberOfValidation = 0
    numberOfTest = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        fc_counter = corrcoef(counter["counter_timeseries"])

        fc_counter = fc_counter[np.triu_indices_from(fc_counter, k=1)]

        # print("counter['subjId'] = ", counter["subjId"])

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                numberOfValidation += 1
                continue
            numberOfTest += 1

        targetFCs.append(fc_counter.flatten())

    targetFCs = np.array(targetFCs)

    if (targetDataset == "hcpRest_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpRest/hcpRest_fcArray.save"
        )
    elif (targetDataset == "hcpTask_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpTask/hcpTask_fcArray.save"
        )
    elif (targetDataset == "id1000_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/id1000/id1000_fcArray.save"
        )

    sourceFCs = loaded[0]
    sourceSubjIds = loaded[1]

    sourceFCs = sourceFCs.reshape(sourceFCs.shape[0], -1)

    batchSize = 8

    if (targetDataset == "hcpTask_0"):
        batchSize = 1

    nOfSourceSamples = sourceFCs.shape[0]
    nOfTargetSamples = targetFCs.shape[0]

    runningSimilarity = 0
    j = 0

    for i in tqdm(range(0, nOfTargetSamples, batchSize), ncols=60, ascii=True):

        left = i
        right = min(i + batchSize, nOfTargetSamples)

        j += (right - left)

        similarity = cosine_similarity(targetFCs[left:right], sourceFCs)

        runningSimilarity += np.mean(similarity) * (right - left)

    runningSimilarity /= j

    return runningSimilarity


def calculateWassersteinDistanceOfFCs(counterBag, targetDataset, isVal,
                                      validationSubjIds):
    """
        targetFCs: (nOfSamples, nOfNodes, nOfNodes)
    """

    targetFCs = []
    targetSubjIds = []

    # print("validationSubjIds = ", validationSubjIds)
    print("isVal = ", isVal)

    numberOfValidation = 0
    numberOfTest = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        if(not counter["success"]):
            continue

        fc_counter = corrcoef(counter["counter_timeseries"])

        fc_counter = fc_counter[np.triu_indices_from(fc_counter, k=1)]

        # print("counter['subjId'] = ", counter["subjId"])

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                numberOfValidation += 1
                continue
            numberOfTest += 1

        originalLabel = counter["originalLabel"]
        if isinstance(originalLabel, torch.Tensor):
            originalLabel = originalLabel.numpy()
        originalLabel = str(originalLabel)

        targetFCs.append(fc_counter.flatten())
        targetSubjIds.append(counter["subjId"] + "_" + originalLabel)

    print("numberOfValidation = ", numberOfValidation)
    print("numberOfTest = ", numberOfTest)

    targetFCs = np.array(targetFCs)

    if (targetDataset == "hcpRest_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpRest/hcpRest_fcArray.save"
        )
    elif (targetDataset == "hcpTask_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpTask/hcpTask_fcArray.save"
        )
    elif (targetDataset == "id1000_0"):
        loaded = torch.load(
            "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/id1000/id1000_fcArray.save"
        )

    sourceFCs = loaded[0]
    sourceSubjIds = loaded[1]

    sourceFCs = sourceFCs.reshape(sourceFCs.shape[0], -1)

    batchSize = 8

    if (targetDataset == "hcpTask_0"):
        batchSize = 4

    nOfSourceSamples = sourceFCs.shape[0]
    nOfTargetSamples = targetFCs.shape[0]

    distances = np.zeros((nOfTargetSamples, nOfSourceSamples))

    print("sourceFCs.shape = ", sourceFCs.shape)
    print("targetFCs.shape = ", targetFCs.shape)

    for i in tqdm(range(0, nOfTargetSamples, batchSize), ncols=60, ascii=True):

        left = i
        right = min(i + batchSize, nOfTargetSamples)

        distances[left:right, :] = np.sqrt(
            -2 * np.dot(targetFCs[left:right], sourceFCs.T) +
            np.sum(sourceFCs**2, axis=1) +
            np.sum(targetFCs[left:right]**2, axis=1)[:, np.newaxis] + 1e-8)

        # print(
        #     "distances[left:right, :].shape = ",
        #     np.sqrt(
        #         -2 * np.dot(targetFCs[left:right], sourceFCs.T) +
        #         np.sum(sourceFCs**2, axis=1) +
        #         np.sum(targetFCs[left:right]**2, axis=1)[:, np.newaxis]).shape)

    weights1 = np.ones(nOfTargetSamples) / nOfTargetSamples
    weights2 = np.zeros(nOfSourceSamples)
    weights2[:] = 0  # 1 / (nOfSourceSamples - nOfTargetSamples)

    # print("targetSubjIds = ", targetSubjIds)
    # print("sourceSubjIds = ", sourceSubjIds)

    print("targetSubjdIds[0] in sourceSubjIds", targetSubjIds[0]
          in sourceSubjIds)

    unique_subjIds = set(targetSubjIds)
    print("len(unique_subjIds) = ", len(unique_subjIds))
    print("len(targetSubjIds) = ", len(targetSubjIds))

    for subjId in targetSubjIds:
        indexOfSubjId = sourceSubjIds.index(subjId)
        weights2[indexOfSubjId] = 1 / nOfTargetSamples

    print("np.sum(weights2 == 1/nOfTargetSamples) = ",
          np.sum(weights2 == 1 / nOfTargetSamples))
    # weights2 /= 2

    print("weights1.shape = ", weights1.shape)
    print("weights2.shape = ", weights2.shape)
    print("distances.shape = ", distances.shape)

    wasserstein_distance = ot.emd2(weights1, weights2, distances)

    return wasserstein_distance


def calculateFID(targetSampleDirectory,
                 targetDataset,
                 device,
                 isVal,
                 validationSubjIds,
                 isFc=False):

    if (targetDataset == "hcpRest_0"):
        if (not isFc):
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpRest/images"
        else:
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpRest/images_fc"

    if (targetDataset == "hcpTask_0"):
        if (not isFc):
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpTask/images"
        else:
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/hcpTask/images_fc"

    if (targetDataset == "id1000_0"):

        if (not isFc):
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/id1000/images"
        else:
            sourceSampleDirectory = "/auto/data2/abedel/data/Projects/Counterfact/Analysis/FID/Images/id1000/images_fc"

    print("targetSampleDirectory = {}".format(targetSampleDirectory))
    print("sourceSampleDirectory = {}".format(sourceSampleDirectory))

    if (targetDataset == "hcpTask_0"):
        windowCropSize = 100
        batchSize = 1
    else:
        windowCropSize = 0
        batchSize = 8

    m1, s1 = compute_statistics_of_path_(path=targetSampleDirectory,
                                         batch_size=batchSize,
                                         device=device,
                                         dims=2048,
                                         isVal=isVal,
                                         validationSubjIds=validationSubjIds,
                                         windowCropSize=windowCropSize)

    m2, s2 = compute_statistics_of_path_(path=sourceSampleDirectory,
                                         batch_size=batchSize,
                                         device=device,
                                         dims=2048,
                                         isVal=None,
                                         validationSubjIds=None,
                                         windowCropSize=windowCropSize)

    fid1 = calculate_frechet_distance(m1, s1, m2, s2)
    fid2 = calculate_frechet_distance(m2, s2, m1, s1)
    print("fid1 = {}, fid2 = {}".format(fid1, fid2))

    return fid1, fid2


def calculateFlipRate(counterBag, targetClass, isVal, validationSubjIds):

    flipRate = 0.0

    count = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                continue

        count += 1

        if (counter["success"]):

            flipRate += 1

    return flipRate / count


def calculateMeanProbability(counterBag,
                             targetClass,
                             isVal,
                             validationSubjIds,
                             isChex=False):

    meanProbability = 0.0

    count = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                continue

        if (not counter["success"]):
            continue

        count += 1

        if ("prob_after" not in counter):
            meanProbability += counter["prob_after_for_targetClass_{}".format(
                targetClass)][targetClass]
        else:
            prob_after = counter["prob_after"]
            # print("targetClass = {}".format(targetClass))
            # print("prob_after = {}".format(prob_after))
            # print("success = ", counter["success"])
            # print("counterLabel = ", counter["counterLabel"])
            # print("targetClass = ", counter["targetClass"])
            if (prob_after.ndim == 1):
                meanProbability += prob_after[targetClass]
            else:
                meanProbability += prob_after[0][targetClass]

    return meanProbability / count


def calculateProximity(counterBag, isVal, validationSubjIds, isChex=False):

    running_proximity = 0.0
    running_proximity_fc = 0.0

    count = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                continue

        if (not counter["success"]):
            continue

        count += 1

        timeseries = counter["timeseries"]
        counter_timeseries = counter["counter_timeseries"]

        if isinstance(timeseries, torch.Tensor):
            timeseries = timeseries.numpy()

        if isinstance(counter_timeseries, torch.Tensor):
            counter_timeseries = counter_timeseries.numpy()

        fc = corrcoef(timeseries)
        fc_counter = corrcoef(counter_timeseries)

        mse = torch.nn.functional.mse_loss(
            torch.from_numpy(timeseries).float(),
            torch.from_numpy(counter_timeseries).float())

        running_proximity += mse

        running_proximity_fc += torch.nn.functional.mse_loss(
            torch.from_numpy(fc).float(),
            torch.from_numpy(fc_counter).float())

    return running_proximity / count, running_proximity_fc / count


def calculateSparsity(counterBag, isVal, validationSubjIds, isChex=False):

    # std of input data

    totalActivationPoints = 0.0
    totalActivationPoints_fc = 0.0

    count = 0

    for counter in counterBag:

        if (not counter["isTest"]):
            continue

        if (isVal):
            if (counter["subjId"] not in validationSubjIds):
                continue
        else:
            if (counter["subjId"] in validationSubjIds):
                continue

        if (not counter["success"]):
            continue

        count += 1

        timeseries = counter["timeseries"]
        counter_timeseries = counter["counter_timeseries"]

        if isinstance(timeseries, torch.Tensor):
            timeseries = timeseries.numpy()

        if isinstance(counter_timeseries, torch.Tensor):
            counter_timeseries = counter_timeseries.numpy()

        fc = corrcoef(timeseries)
        fc_counter = corrcoef(counter_timeseries)

        threshold = np.std(timeseries)
        threshold_fc = np.std(fc)

        thresholded_diff = np.abs(timeseries - counter_timeseries) > threshold

        thresholded_diff_fc = np.abs(fc - fc_counter) > threshold_fc

        totalActivationPoints += np.mean(thresholded_diff.astype(np.float32))
        totalActivationPoints_fc += np.mean(
            thresholded_diff_fc.astype(np.float32))

    return totalActivationPoints / count, totalActivationPoints_fc / count


def calculateDiversity(counterBag):
    pass
