import numpy as np


def dummyLoader():

    nOfClasses_dummy = 2
    N_dummy = 400  # number of rois
    T_dummy = 300  # scan length

    dummy_subjectCount = 100
    subjectDatas_dummy = []
    subjectIds_dummy = []
    labels_dummy = []

    for i in range(dummy_subjectCount):

        subjectId = i

        subjectData = np.random.randn(N_dummy, T_dummy)

        subjectDatas_dummy.append(subjectData)
        subjectIds_dummy.append(subjectId)
        labels_dummy.append(np.random.randint(nOfClasses_dummy))

    return subjectDatas_dummy, labels_dummy, subjectIds_dummy
