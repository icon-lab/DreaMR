from Utils.utils import Option
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

from .DataLoaders.dummyLoader import dummyLoader

loaderMapper = {
    "dummy": dummyLoader,
}


def clipData(x):

    max = 6.0
    min = -6.0

    array = np.clip(x, min, max)
    return array


class SupervisedDataset(Dataset):

    def __init__(self, dataset, dynamicLength, batchSize, foldCount):

        datasetName = dataset

        self.batchSize = batchSize
        self.dynamicLength = dynamicLength
        self.foldCount = foldCount

        self.seed = 0

        loader = loaderMapper[datasetName]

        self.kFold = StratifiedKFold(foldCount,
                                     shuffle=False,
                                     random_state=None)
        self.k = None

        self.data, self.labels, self.subjectIds = loader()

        self.inputDim = self.data[0].shape[0]  # assuming of shape (R, T)
        self.nOfClasses = np.max(self.labels) + 1

        random.Random(self.seed).shuffle(self.data)
        random.Random(self.seed).shuffle(self.labels)
        random.Random(self.seed).shuffle(self.subjectIds)

        self.targetData = None
        self.targetLabel = None
        self.targetSubjIds = None
        self.validationSubjIds = None

        self.randomRanges = None

        self.trainIdx = None
        self.testIdx = None
        self.valIdx = None

    def __len__(self):
        if (self.isGenerating):
            return len(self.targetDataGroups[self.targetLengthGroup])
        else:
            return len(self.targetData)

    def normalizeListOfData(self, data):
        normalizedData = []

        for roiSignal in data:
            roiSignal = (roiSignal - np.mean(roiSignal, axis=1, keepdims=True)
                         ) / np.std(roiSignal, axis=1, keepdims=True)
            roiSignal = np.nan_to_num(roiSignal, 0)

            roiSignal = clipData(roiSignal)

            normalizedData.append(roiSignal)

        return normalizedData

    def setFold(self, fold, train=True):

        self.k = fold
        self.train = train
        self.isGenerating = False
        self.targetClass = None

        #fold = range(5)[4-fold]

        if (self.foldCount == None):  # if this is the case, train must be True
            trainIdx = list(range(len(self.data)))
            testIdx = None
            valIdx = None
        else:
            trainIdx, testIdx = list(self.kFold.split(self.data,
                                                      self.labels))[fold]

            random.Random(self.seed).shuffle(testIdx)
            valIdx = testIdx[:len(testIdx) // 2]

        self.trainIdx = trainIdx
        self.testIdx = testIdx
        self.valIdx = valIdx

        random.Random(self.seed).shuffle(trainIdx)

        self.targetSubjIds = [
            self.subjectIds[idx] for idx in trainIdx
        ] if train else [self.subjectIds[idx] for idx in testIdx]

        self.validationSubjIds = [self.subjectIds[idx] for idx in valIdx]

        self.targetData = [self.data[idx] for idx in trainIdx
                           ] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [
            self.labels[idx] for idx in trainIdx
        ] if train else [self.labels[idx] for idx in testIdx]

        np.random.seed(self.seed + 1)

        if (not isinstance(self.dynamicLength, type(None))):
            self.randomRanges = [[
                np.random.randint(
                    0, self.targetData[idx].shape[-1] - self.dynamicLength)
                for k in range(8 * 200)
            ] for idx in range(len(self.targetData))]

    def getFold(self, fold, train=True):

        self.setFold(fold, train)

        if (train):
            return DataLoader(self,
                              batch_size=self.batchSize,
                              shuffle=False,
                              pin_memory=False)
        else:
            return DataLoader(self,
                              batch_size=1,
                              shuffle=False,
                              pin_memory=False)

    def setFold_gen(self, fold, targetClass, train=True):

        self.k = fold
        self.train = train
        self.isGenerating = True
        self.targetClass = targetClass

        trainIdx_, testIdx_ = list(self.kFold.split(self.data,
                                                    self.labels))[fold]

        trainIdx = []
        testIdx = []

        for idx in trainIdx_:
            if (self.labels[idx] != targetClass):
                trainIdx.append(idx)

        for idx in testIdx_:
            if (self.labels[idx] != targetClass):
                testIdx.append(idx)

        self.trainIdx = trainIdx
        self.testIdx = testIdx

        self.targetData = [self.data[idx] for idx in trainIdx
                           ] if train else [self.data[idx] for idx in testIdx]
        self.targetLabels = [
            self.labels[idx] for idx in trainIdx
        ] if train else [self.labels[idx] for idx in testIdx]

        self.targetSubjIds = [
            self.subjectIds[idx] for idx in trainIdx
        ] if train else [self.subjectIds[idx] for idx in testIdx]

        # group data by their length

        self.targetDataGroups = {}
        self.targetLabelGroups = {}
        self.targetSubjIdGroups = {}

        for idx in range(len(self.targetData)):

            length = self.targetData[idx].shape[-1]

            if (length not in self.targetDataGroups):
                self.targetDataGroups[length] = []
                self.targetLabelGroups[length] = []
                self.targetSubjIdGroups[length] = []

            self.targetDataGroups[length].append(self.targetData[idx])
            self.targetLabelGroups[length].append(self.targetLabels[idx])
            self.targetSubjIdGroups[length].append(self.targetSubjIds[idx])

        self.randomRanges = None

        self.lengthGroups = list(self.targetDataGroups.keys())

    def getSet_gen(self, targetLengthGroup, batchSize_gen):

        self.targetLengthGroup = targetLengthGroup

        return DataLoader(self,
                          batch_size=batchSize_gen,
                          shuffle=False,
                          pin_memory=False)

    def __getitem__(self, idx):

        if (self.isGenerating):

            scan = self.targetDataGroups[self.targetLengthGroup][idx]
            label = self.targetLabelGroups[self.targetLengthGroup][idx]
            subjId = self.targetSubjIdGroups[self.targetLengthGroup][idx]

        else:

            scan = self.targetData[idx]
            label = self.targetLabels[idx]
            subjId = self.targetSubjIds[idx]

        timeseries = scan  # (numberOfRois, time)

        timeseries = (timeseries - np.mean(timeseries, axis=1, keepdims=True)
                      ) / np.std(timeseries, axis=1, keepdims=True)
        timeseries = np.nan_to_num(timeseries, 0)

        timeseries = clipData(timeseries)

        if (not self.isGenerating):

            if (self.train and not isinstance(self.dynamicLength, type(None))):
                if (timeseries.shape[1] < self.dynamicLength):
                    print(timeseries.shape[1], self.dynamicLength)

                samplingInit = self.randomRanges[idx].pop()

                timeseries = timeseries[:, samplingInit:samplingInit +
                                        self.dynamicLength]

        return {
            "timeseries": timeseries.astype(np.float32),
            "label": label,
            "subjId": subjId
        }
