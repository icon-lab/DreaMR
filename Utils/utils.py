from genericpath import exists
from sklearn import metrics as skmetr
from datetime import datetime
import numpy as np
import copy
from glob import glob
import os
import torch

from PIL import Image
from matplotlib import cm

import importlib

from Utils.variables import resultsSaveDir


class Option(object):

    def __init__(self, my_dict):

        self.dict = my_dict

        for key in my_dict:

            if (key == "dict"):
                continue

            if isinstance(my_dict[key], dict):

                setattr(self, key, Option(my_dict[key]))

            else:
                setattr(self, key, my_dict[key])

    def setattr_(self, key, value):

        setattr(self, key, value)

        if isinstance(value, Option):
            self.dict[key] = getattr(self, key).dict
        else:
            self.dict[key] = value

    def getattr_(self, key):
        return getattr(self, key)

    def copy(self):
        return Option(copy.deepcopy(self.dict))


def writeOptionsToFile(options, file):

    print("From write option to file and options : {}".format(options))

    for item in list(vars(options).items()):

        if (item[0] == "dict"):
            continue

        if isinstance(item[1], Option):

            file.write("{} : *** [\n".format(item[0]))

            for item_ in list(vars(item[1]).items()):

                if (item_[0] == "dict"):
                    continue

                file.write("    {} : {}\n".format(item_[0], item_[1]))

            file.write("] ***\n")

        elif isinstance(item[1], dict):

            file.write("{} : *** [ \n ".format(item[0]))

            for key in item[1]:
                file.write("    {} : {}\n".format(key, item[1][key]))

            file.write("]***\n")

        else:

            file.write("{} : {}\n".format(item[0], item[1]))


def writeDictToFile(content, file, level=0):

    for item in list(content.keys()):

        prepend = ""
        for i in range(level):
            prepend += " "
        file.write("\n{}{}:".format(prepend, item))

        if (isinstance(content[item], dict)):
            writeDictToFile(content[item], file, level + 1)
        else:
            file.write("{}".format(content[item]))

    if (level == 0):
        file.write("\n\n")


def overWriteActions(hyperParams_, activations):

    hyperParams = hyperParams_.copy()

    print("activations = {}".format(activations))

    for item in list(vars(activations).items()):

        if (item[0] == "dict"):
            continue

        if isinstance(item[1], Option):
            hyperParams.setattr_(
                item[0],
                overWriteActions(hyperParams.getattr_(item[0]), item[1]))

        else:
            hyperParams.setattr_(item[0], item[1])

    return hyperParams


def getRecentGenPath(runPath, fold):

    gens = glob(runPath + "/fold{}/CounterSamples".format(fold) + "/gen-*")

    def getDateAsNumber(year, month, day, hour, minute, second):
        return year * 365 * 24 * 60 * 60 + month * 30 * 24 * 60 * 60 + day * 24 * 60 * 60 + \
            hour * 60 * 60 + minute * 60 + second

    def genNameToDateAsNumber(genName):

        startOffset = 4

        year = int(genName.split("-")[1 + startOffset])
        month = int(genName.split("-")[2 + startOffset])
        day = int(genName.split("-")[3 + startOffset])
        hour = int(genName.split("-")[4 + startOffset].split(":")[0])
        minute = int(genName.split("-")[4 + startOffset].split(":")[1])
        second = int(genName.split("-")[4 + startOffset].split(":")[2])

        return getDateAsNumber(year, month, day, hour, minute, second)

    gens = sorted(gens, key=lambda x: genNameToDateAsNumber(x))

    print("target gen = ", gens[-1])

    return gens[-1]


def getRecentRunPath(method, targetDataset):

    runs = glob(resultsSaveDir +
                "/Counterfacts/{}/{}/run-*".format(method, targetDataset))

    def getDateAsNumber(year, month, day, hour, minute, second):
        return year * 365 * 24 * 60 * 60 + month * 30 * 24 * 60 * 60 + day * 24 * 60 * 60 + \
            hour * 60 * 60 + minute * 60 + second

    def runNameToDateAsNumber(runName):
        year = int(runName.split("-")[1])
        month = int(runName.split("-")[2])
        day = int(runName.split("-")[3])
        hour = int(runName.split("-")[4].split(":")[0])
        minute = int(runName.split("-")[4].split(":")[1])
        second = int(runName.split("-")[4].split(":")[2])

        return getDateAsNumber(year, month, day, hour, minute, second)

    runs = sorted(runs, key=lambda x: runNameToDateAsNumber(x))

    print("runs = {}".format(runs[-1]))

    return runs[-1]


def loadModel_distill_expert(modelPath, timesteps, expertIndex, device):
    print("* -- * loading for device = {}".format(device))
    modelFiles = glob(modelPath +
                      "/Expert_{}/model_distill_{}_expert_{}_epoch_*".format(
                          expertIndex, timesteps, expertIndex))
    modelFiles = sorted(modelFiles,
                        key=lambda x: int(x.split("_")[-1].split(".")[0]))
    if (modelFiles != None and len(modelFiles) > 0):
        print("loading model, " + modelFiles[-1])
        model = torch.load(modelFiles[-1], map_location=device)

        epoch = int(modelFiles[-1].split("_")[-1].split(".")[0])
        return model, epoch
    return None


def saveModel_distill_expert(model, modelPath, timesteps, epoch, expertIndex):

    expertFolder = modelPath + "/Expert_{}".format(expertIndex)
    os.makedirs(expertFolder, exist_ok=True)

    modelFiles = glob(
        expertFolder +
        "/model_distill_{}_expert_{}_epoch_*".format(timesteps, expertIndex))

    coolEpochsList = [25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 800]
    for modelFile in modelFiles:
        itMatches = False
        for coolEpoch in coolEpochsList:
            if ("model_distill_{}_expert_{}_epoch_{}.save".format(
                    timesteps, expertIndex, coolEpoch) in modelFile):
                itMatches = True
                break
        if (not itMatches):
            os.remove(modelFile)

    torch.save(
        model,
        expertFolder + "/model_distill_{}_expert_{}_epoch_{}.save".format(
            timesteps, expertIndex, epoch))


def writeParamsLog(fileName, paramDict):
    file_ = open(fileName, "w")
    file_.write("\n")
    writeDictToFile(paramDict, file_, 0)
    file_.write("\n\n")
    file_.close()


def writeMessageLog(fileName, logMessage):

    file_ = open(fileName, "a")
    file_.write("\n")
    file_.write(logMessage)
    file_.write("\n")
    file_.close()


def arrayToPILImage(array):

    # normalize to 0 and 1
    array += np.abs(np.min(array))
    array /= np.max(array)

    # apply colormap here
    imageArray = cm.magma(array)
    image = Image.fromarray(np.uint8(imageArray * 255))

    return image
