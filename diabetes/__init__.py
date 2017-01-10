from diabetes import Diabetes

#local imports
from conf import *

class Diabetic():
    def __init__(self):
        self.diabetes = Diabetes()

    def save(self, location):
        self.diabetes.save(file_name=location)

    def load(self, location):
        self.diabetes.load(file_name=location)

    def train(self):
        self.diabetes.addTrainingData()
        self.diabetes.train()

    def test(self,input_data):
        result = self.diabetes.classify(input_data)
        print(result)

    # def classify(self):
    #     self.diabetes.classify()

    def accuracy(self):
        self.diabetes.accuracy()

    def classify(self):
        total = 0
        right = 0
        wrong = 0

        for row in read_csv():
            total += 1
            attributes = list(map(float, row[:8]))

            if (row[8] == "1"):
                result = self.diabetes.classify(attributes)
                if result == 1:
                    right += 1
                else:
                    wrong += 1
                print "ACTUAL: Diabetic"
            else:
                result = self.diabetes.classify(attributes)
                if result == 0:
                    right += 1
                else:
                    wrong += 1
                print "ACTUAL: Not Diabetic"

        print "total: %4d" % total, \
              "right: %4d" % right, \
              "wrong: %4d" % wrong
