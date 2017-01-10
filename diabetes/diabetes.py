import csv
import os
import sys
from sys import argv
import pickle

#local imports
from conf import *

#PyBrain imports
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain import TanhLayer,SigmoidLayer
from pybrain.structure.modules import SoftmaxLayer

class Diabetes():
    def __init__(self, hidden_nodes = 30):
        self.network = buildNetwork(8, hidden_nodes, 2, hiddenclass=SigmoidLayer, bias=True)
        self.data_set = ClassificationDataSet(8, 1, nb_classes=2)
        self.train_data_set = ClassificationDataSet(8, 1, nb_classes=2)
        self.test_data_set = ClassificationDataSet(8, 1, nb_classes=2)
        self.trainer = BackpropTrainer(self.network, self.data_set, verbose=True)

    def addTrainingData(self):
    	with open('/home/sushil/AI/diabetes_person_test/data/diabetes.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                data = [float(x) for x in row if x != '']
                self.data_set.addSample(data[:8], [data[8]])

    def train(self):
        self.data_set._convertToOneOfMany()
        #self.train_data_set, self.test_data_set = self.data_set.splitWithProportion(0.75)
        self.trainer.trainEpochs(30)
        # self.trainer.trainUntilConvergence()

    def save(self, file_name="classifier.diabetes"):
        with open(get_path(file_name), 'wb') as file_pointer:
            pickle.dump(self.network, file_pointer)

    def load(self, file_name="classifier.diabetes"):
        with open(get_path(file_name), 'rb') as file_pointer:
            self.network = pickle.load(file_pointer)

    def accuracy(self):
        if len(self.data_set) == 0:
            print "No data_sets found. Maybe you loaded the classifier from a file?"
            return

        tstresult = percentError(
                        self.trainer.testOnClassData(dataset=self.data_set),
                        self.data_set['class']
                    )

        print "epoch: %4d" % self.trainer.totalepochs, \
              "trainer error: %5.2f%%" % tstresult, \
              "trainer accuracy: %5.2f%%" % (100-tstresult)


    def classify(self,test_data):
            score = self.network.activate(test_data)
            print(score)

            score = max(range(len(score)), key=score.__getitem__)
            print(score)
            if score == 0:
                return 0
            else:
                return 1


# def main():
#     filename = 'data/diabetes.csv'
#     dataset = loadCsv(filename)
#     datasetclass = loadClassCsv(filename)
#     print(dataset[:2])
#     print(datasetclass[:2])
#     #print('Loaded data file {0} with {1} rows').format(filename, len(dataset))
# main()
