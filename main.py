from CounterModels.DreaMR.run_counter_dreamr import train_counter_dreamr, gen_counter_dreamr, eval_counter_dreamr
from Classifiers.BolT.run_classifier_bolt import run_bolT
import argparse
from Utils.gpuChecker import getAvailableGpus
from datetime import datetime
from Utils.utils import Option

parser = argparse.ArgumentParser()


parser.add_argument("--targetDataset", type=str, default="dummy")
parser.add_argument("--method", type=str, default="dreamr")
parser.add_argument("--loadThreshold", type=str, default=0.5)
parser.add_argument("--do", type=str, default="train")
parser.add_argument("--fromExists", type=int, default=0)
parser.add_argument("--isVal", type=int, default=0)

argv = parser.parse_args()

availableGpus = []
while (len(availableGpus) == 0):
    availableGpus = getAvailableGpus(float(argv.loadThreshold))
gpu = availableGpus[0]  # deneme
device = "cuda:{}".format(gpu)

foldCount = 5  # ignore this
nOfClasses = 2
dynamicLength = 128
datePrepend = "{}".format(datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))

targetClassifierPath = "your/target/classifier/path/model.pt"

targetRunFolder = None  # targetRunFolder: "your/target/run/folder"
targetGenFolders = None
# targetGenFolders : ["your/target/gen/folder1", "your/target/gen/folder2"]


details = Option({
    "device": device,
    "foldCount": foldCount,
    "datePrepend": datePrepend,
    "targetDataset": argv.targetDataset,
    "classifierPath": targetClassifierPath,
    "fromExists": argv.fromExists,
    "nOfClasses": nOfClasses,
    "dynamicLength": dynamicLength,
    "targetRunFolder": None,
    "targetGenFolders": None,
    "methodName": argv.method,
    "isVal": argv.isVal
})

# classifiers
# counterfactual models


trainers = {
    # classifier trainers
    "bolT_classify": run_bolT,
    # counter trainers
    "dreamr": train_counter_dreamr,
}

generators = {
    "defacto": gen_counter_dreamr,
}

evaluaters = {
    "defacto": eval_counter_dreamr,
}


if ("train" in argv.do):
    trainer = trainers[argv.method]
    trainer(details)

if ("gen" in argv.do):
    generator = generators[argv.method]
    generator(details)

if ("eval" in argv.do):
    evaluater = evaluaters[argv.method]
    evaluater(details)
