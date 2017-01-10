from diabetes import Diabetes

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
