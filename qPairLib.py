import numpy as np


# TODO: isolate by time, isolate by time delay


class PairSet:
    def __init__(self, dataset, label='None'):
        if not isinstance(dataset, np.ndarray):
            raise TypeError("Dataset needs to be a numpy ndarray")

        if len(dataset.shape) != 2 or dataset.shape[1] != 2:
            raise TypeError("Dataset needs to be a numpy ndarray of shape (x,2) ")

        self.label = label
        self.data = dataset

    @classmethod
    def from_file(cls, filename):
        dataset = np.load(filename)
        return cls(dataset)

    def generateTriplet(self, pairset, label='None'):

        triplet_t1 = []
        triplet_t2 = []
        triplet_t3 = []

        for i in range(len(self.data)):

            t1 = self.data[i, 0]
            dt_12 = self.data[i, 1]

            j1 = np.searchsorted(pairset.data[:, 0], t1, side='left')
            j2 = np.searchsorted(pairset.data[:, 0], t1, side='right')

            if j1 != j2:  # Entry is present
                triplet_t1.append(t1)
                triplet_t2.append(t1 + dt_12)
                triplet_t3.append(t1 + pairset.data[j1, 1])

            if i % 10000000 == 0 and i != 0:
                print("i=", i, " pairs_123.size=", len(triplet_t1))

        triplet_dataset = np.column_stack((triplet_t1, triplet_t2, triplet_t3))
        return TripletSet(triplet_dataset, label)

    def filter_dt(self, dt1, dt2):
        dt_window = np.logical_and(self.data[:, 1] > dt1, self.data[:, 1] <= dt2)
        return PairSet(self.data[dt_window])


class TripletSet:
    def __init__(self, dataset, label='None'):
        if not isinstance(dataset, np.ndarray):
            raise TypeError("Dataset must be a numpy ndarray")

        if len(dataset.shape) != 2 or dataset.shape[1] != 3:
            raise TypeError("Dataset must be a numpy ndarray of shape (x,3) ")

        self.label = label
        self.data = dataset
