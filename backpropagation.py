#Yishan Li 10182827
import numpy as np
import math
from tensorflow.keras.datasets import mnist
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import time

class backpropagation:
    def __init__(self):
        self.numberOfHiddenNode = 80
        self.numberOfInputNode = 785
        self.numberOfTrainingData = 60000
        self.numberOfTestingData = 10000
        self.outputNode = 10
        self.epoch = 20
        self.learningRate = 0.9
        self.momentum = 0.65
        self.weightLayer1 = []
        self.weightLayer2 = []
        self.inputTrainingData,self.desireTrainingData,self.inputTestingData, self.desireTestingData = self.readData()
        self.trainingDesireTemp = []

        self.inputFromHiddenToOutputLayer = []
        self.activationLayer1 = []
        self.activationLayer2 = []
        self.outputFromOutputLayer = []

        self.deltaWeightFromHiddenToInput = []
        self.deltaWeightFromHiddenToInputPreviousEpoch = []

        self.deltaWeightFromOutputToHidden = []
        self.deltaWeightFromOutputToHiddenPreviousEpoch = []

        self.totalErrorForTrainingData = 0
        self.totalErrorForTrainingDataPreviousEpoch = 0

        self.totalErrorForTestingData = 0
        self.totalErrorThresholdForTestingData = 0.07

    def readData(self):
        # Load the training data and testing data.
        (X_train,y_train),(X_test,y_test) = mnist.load_data()
        return X_train, y_train, X_test, y_test

    # each image is a multi-array
    def preprocessRawData(self, dataset):
        # Turn the multi-array to a single array
        inputData = []
        for i in range(len(dataset)):# 60000 training data in total
            eachDataEntry = []
            for j in range(28):
                eachDataEntry = np.concatenate((eachDataEntry, dataset[i][j]))
            eachDataEntry = np.concatenate((eachDataEntry,np.asarray([1])))
            inputData.append(eachDataEntry)
        return np.asarray(inputData)

    # Scale the input data to range 0-1.
    def normalizeData(self,dataset):
        for i in range(len(dataset)):# 60000 training data in total
            for j in range(self.numberOfInputNode):# 785 input values for each dataset
                dataset[i][j] = dataset[i][j]/255 #Divided by 255, since 255 is the maximum value within the set
        return dataset

    # Change the desired output digit to one-hot vector for training purpose.
    def changeDesiredOutputFormat(self, desireOutput):
        newDesireOutputFormat = []
        for i in range(desireOutput.size):
            if desireOutput[i] == 0:
                newDesireOutputFormat.append([1,0,0,0,0,0,0,0,0,0])
            if desireOutput[i] == 1:
                newDesireOutputFormat.append([0,1,0,0,0,0,0,0,0,0])
            if desireOutput[i] == 2:
                newDesireOutputFormat.append([0,0,1,0,0,0,0,0,0,0])
            if desireOutput[i] == 3:
                newDesireOutputFormat.append([0,0,0,1,0,0,0,0,0,0])
            if desireOutput[i] == 4:
                newDesireOutputFormat.append([0,0,0,0,1,0,0,0,0,0])
            if desireOutput[i] == 5:
                newDesireOutputFormat.append([0,0,0,0,0,1,0,0,0,0])
            if desireOutput[i] == 6:
                newDesireOutputFormat.append([0,0,0,0,0,0,1,0,0,0])
            if desireOutput[i] == 7:
                newDesireOutputFormat.append([0,0,0,0,0,0,0,1,0,0])
            if desireOutput[i] == 8:
                newDesireOutputFormat.append([0,0,0,0,0,0,0,0,1,0])
            if desireOutput[i] == 9:
                newDesireOutputFormat.append([0,0,0,0,0,0,0,0,0,1])
        return newDesireOutputFormat

    # Initialize the weight vector.
    def initializeWeight(self):
        # The input layer to hidden layer's initial weight
        self.weightLayer1 = np.random.uniform(-1, 1, size=(785, self.numberOfHiddenNode))
        # The hidden layer to output layer's initial weight
        self.weightLayer2 = np.random.uniform(-1, 1, size=(self.numberOfHiddenNode, 10))

    # Apply sigmoid function each node in hidden layer and output layer.
    def outputFuction(self,dataset):
        outputFromHiddenLayer = []
        for value in dataset:
            output = 1 / (1 + (math.e ** (-value)))
            outputFromHiddenLayer.append(output)
        return np.asarray(outputFromHiddenLayer)

    # The delta weight between hidden layer and output layer need updating.
    def deltaWeightFromOutputLayerToHiddenLayer(self, whichDataInput):
        errorAndDerivative = []
        deltaWeight = []
        derivative = []
        error = []
        for i in range(self.outputNode):
            # add up the total error for each of the output node within each epoch
            self.totalErrorForTrainingData = self.totalErrorForTrainingData + (self.desireTrainingData[whichDataInput][i]-self.outputFromOutputLayer[i])**2
        # Get the error times the derivative result at the output layer.
        error = np.asarray(self.desireTrainingData[whichDataInput]) - np.asarray(self.outputFromOutputLayer)
        derivative = np.multiply(np.asarray(self.outputFromOutputLayer),(np.ones(10) - np.asarray(self.outputFromOutputLayer)))
        errorAndDerivative = self.learningRate * np.multiply(error,derivative)
        deltaWeight =np.multiply(np.expand_dims(np.transpose(np.asarray(errorAndDerivative)),1), self.inputFromHiddenToOutputLayer)
        self.deltaWeightFromOutputToHiddenPreviousEpoch = self.deltaWeightFromOutputToHidden.copy()
        self.deltaWeightFromOutputToHidden = np.transpose(np.asarray(deltaWeight))

    # The delta weight between input layer and hidden layer that need updating.
    def deltaWeightFromHiddenLayerToInputLayer(self, whichDataInput):
        errorAndDerivative = []
        derivative = []
        error = []
        deltaWeightInputToHidden = []
        error = np.asarray(self.desireTrainingData[whichDataInput]) - np.asarray(self.outputFromOutputLayer)
        derivative = np.multiply(np.asarray(self.outputFromOutputLayer),
                                 (np.ones(10) - np.asarray(self.outputFromOutputLayer)))
        errorAndDerivative = self.learningRate * np.multiply(error, derivative)
        layerFromOutputToHidden = np.dot(np.expand_dims(np.asarray(errorAndDerivative),0), np.transpose(self.weightLayer2))
        valueAtEachHiddenNode=np.multiply(np.multiply(self.inputFromHiddenToOutputLayer , (np.ones(self.numberOfHiddenNode) - self.inputFromHiddenToOutputLayer)),layerFromOutputToHidden)
        deltaWeightInputToHidden = np.dot(np.expand_dims(np.transpose(self.inputTrainingData[whichDataInput]),1),
                                          np.asarray(valueAtEachHiddenNode))
        self.deltaWeightFromHiddenToInputPreviousEpoch = self.deltaWeightFromHiddenToInput.copy()
        self.deltaWeightFromHiddenToInput = np.asarray(deltaWeightInputToHidden)

    # Apply the momentum to when updating the weight to avoid local minima
    def backforwardTraining(self):
        if (self.deltaWeightFromOutputToHiddenPreviousEpoch != []):
            self.weightLayer2 = self.weightLayer2 + self.deltaWeightFromOutputToHidden + self.momentum * self.deltaWeightFromOutputToHiddenPreviousEpoch
            self.weightLayer1 = self.weightLayer1 + self.deltaWeightFromHiddenToInput + self.momentum * self.deltaWeightFromHiddenToInputPreviousEpoch
        else: # The first round does not assign the value to variable deltaWeightFromOutputToHiddenPreviousEpoch and deltaWeightFromHiddenToInputPreviousEpoch.
            self.weightLayer2 = self.weightLayer2 + self.deltaWeightFromOutputToHidden
            self.weightLayer1 = self.weightLayer1 + self.deltaWeightFromHiddenToInput

    # Calculate neuron activation at the particular node in either hidden layer or output layer.
    def getActivateValue(self, dataset, weight):
        summationVector = np.dot(dataset, weight)
        return summationVector

    # Before training the model, preprocess the raw data.
    def beforeTrainingModel(self):
        # Put the 28*28 pixel image(multi-array) into one single array, getting 785 inputs(including x0=1).
        self.inputTrainingData = self.preprocessRawData(self.inputTrainingData)
        self.inputTestingData = self.preprocessRawData(self.inputTestingData)

        # Scale the input into the same range(normalize the input)
        self.inputTrainingData = self.normalizeData(self.inputTrainingData)
        self.inputTestingData = self.normalizeData(self.inputTestingData)

        #Inialize the weight vectors
        self.initializeWeight()

        #Change the output to desired format into one hot vector
        self.trainingDesireTemp = self.desireTrainingData
        self.desireTrainingData = self.changeDesiredOutputFormat(self.desireTrainingData)

    # Train the model by going through all the training data.
    def feedforwardTraining(self):
        epoch = 0
        while epoch < self.epoch:
            for i in range(self.numberOfTrainingData):
                if(i % 10000 == 0):
                    print("epoch",epoch, "data i",i)
                # calculate the activation at each of the node at hidden layer
                self.activationLayer1 = self.getActivateValue(self.inputTrainingData[i],self.weightLayer1)
                # Output from hidden layer equals to input from hidden layer to output layer
                self.inputFromHiddenToOutputLayer = self.outputFuction(self.activationLayer1)
                # calculate the activation at each of the node at output layer by feeding the hidden layer output to output layer node
                self.activationLayer2 = self.getActivateValue(self.inputFromHiddenToOutputLayer, self.weightLayer2)
                self.outputFromOutputLayer = self.outputFuction(self.activationLayer2)
                # Calculate the delta weight from hidden layer to output layer
                self.deltaWeightFromOutputLayerToHiddenLayer(i)
                # Calculate the delta weight from input layer to hidden layer
                self.deltaWeightFromHiddenLayerToInputLayer(i)
                self.backforwardTraining()
                self.totalTrainingsetErrorChangingLearningRate()
            # if the total error for testing data is smaller than a threshold, then pause the training
            if(self.calculateTotalErrorForTestingData()):
                break
            epoch += 1

    # Calculate the total error of the testing data set after each epoch
    def calculateTotalErrorForTestingData(self):
        activation1 = []
        inputFromHiddenToOutputLayer = []
        activation2 = []
        testingDataPrediction = []
        self.totalErrorForTestingData = 0
        preprocessTestingData = self.changeDesiredOutputFormat(self.desireTestingData)
        for i in range(self.numberOfTestingData):
            if i % 1000 == 0:
                print("testing data", i)
            activation1 = self.getActivateValue(self.inputTestingData[i], self.weightLayer1)
            # Output from hidden layer equals to input from hidden layer to output layer
            inputFromHiddenToOutputLayer = self.outputFuction(activation1)
            # calculate the activation at each of the node at output layer by feeding the hidden layer output to output layer node
            activation2 = self.getActivateValue(inputFromHiddenToOutputLayer, self.weightLayer2)
            outputFromOutputLayer = self.outputFuction(activation2)
            # Calculated the MSE for testing data using the updated weight
            for j in range(self.outputNode):
                self.totalErrorForTestingData = self.totalErrorForTestingData + (preprocessTestingData[i][j] - outputFromOutputLayer[j])**2
        self.totalErrorForTestingData = self.totalErrorForTestingData / self.numberOfTestingData
        print("totalErrorForTestingset", self.totalErrorForTestingData)

        # If the error is lower than the pre-defined threshold, then we can stop the training.
        if(self.totalErrorForTestingData < self.totalErrorThresholdForTestingData):
            return True
        else:
            return False

    # If the total error of the training data increases, lower the the learning rate.
    # If the total error of the training data decreases, increase the learning rate.
    def totalTrainingsetErrorChangingLearningRate(self):
        self.totalErrorForTrainingData = self.totalErrorForTrainingData / self.numberOfTrainingData
        # Decrease learning rate if the total error is less greater the total error in previous epoch
        if(self.totalErrorForTrainingData > self.totalErrorForTrainingDataPreviousEpoch):
            self.learningRate = 0.999 * self.learningRate
        else:# Increase learning rate if the total error is less than the total error in previous epoch
            self.learningRate = 1.001 * self.learningRate
        self.totalErrorForTrainingDataPreviousEpoch = self.totalErrorForTrainingData
        # continue calculate the total error in the next epoch, so reset the counter
        self.totalErrorForTrainingData = 0

    # Return the predicted vector whichever has shortest distance to the output vector
    def smallestDistance(self, outputFromOutputLayer):
        outputOptions = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],
                         [0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
        dis = [0,0,0,0,0,0,0,0,0,0]
        for i in range(len(outputOptions)):
            dis[i] = self.countDistance(outputFromOutputLayer,outputOptions[i])
        return dis.index(min(dis))

    # Calculate the distance between output data to each of the possible cluster.
    def countDistance(self, vec1, vec2):
        dis = 0
        for i in range(10):
            dis = dis + (vec1[i]-vec2[i])**2
        return math.sqrt(dis)

    # Use the trained model to make prediction upon the testing set.
    def testingData(self, numberOfIteration, inputData,desireOutput):
        numOfSuccess = 0
        prediction = []
        activationLayer1 = []
        inputFromHiddenToOutputLayer = []
        activationLayer2 = []
        outputFromOutputLayer = []
        for i in range(numberOfIteration):
            # Calculate the activation at hidden node.
            activationLayer1 = self.getActivateValue(inputData[i], self.weightLayer1)
            inputFromHiddenToOutputLayer = self.outputFuction(activationLayer1)
            # Calculate the activation at output layer.
            activationLayer2 = self.getActivateValue(inputFromHiddenToOutputLayer, self.weightLayer2)
            outputFromOutputLayer = self.outputFuction(activationLayer2)
            outcome = self.smallestDistance(outputFromOutputLayer)
            prediction.append(outcome)
        return prediction

    # Get the confusion matrix/precision of the testing data
    def defineConfusionMatrixForTestingData(self):
        y_actu = pd.Series(self.desireTestingData, name='Actual')
        y_pred = pd.Series(self.testingData(self.numberOfTestingData, self.inputTestingData,self.desireTestingData), name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        print("Final testing data confusion matrix\n", df_confusion)
        print('Testing dataset Accuracy Score: \n', accuracy_score(self.desireTestingData, self.testingData(self.numberOfTestingData, self.inputTestingData,self.desireTestingData)))
        print('Testing dataset Report : \n', classification_report(self.desireTestingData, self.testingData(self.numberOfTestingData, self.inputTestingData,self.desireTestingData)))

    # Get the confusion matrix/precision of the training data
    def defineConfusionMatrixForTrainingData(self):
        y_actu = pd.Series(self.trainingDesireTemp, name='Actual')
        y_pred = pd.Series(self.testingData(self.numberOfTrainingData, self.inputTrainingData, self.trainingDesireTemp), name='Predicted')
        df_confusion = pd.crosstab(y_actu, y_pred)
        print("Final training data confusion matrix\n", df_confusion)
        print('Training dataset Accuracy Score:\n', accuracy_score(self.trainingDesireTemp, self.testingData(self.numberOfTrainingData, self.inputTrainingData, self.trainingDesireTemp)))
        print('Training dataset Report :\n ', classification_report(self.trainingDesireTemp, self.testingData(self.numberOfTrainingData, self.inputTrainingData, self.trainingDesireTemp)))

    # Put the predicted data to the txt file.
    def writeReport(self):
        # Get all the perdictions of the testing dataset.
        testData = self.testingData(self.numberOfTestingData, self.inputTestingData,self.desireTestingData)
        file = open("result.txt", "w")
        file.write("actual : predicted \n")
        for i in range(len(testData)):
            file.write(str(self.desireTestingData[i]) + "             ")
            file.write(str(testData[i]) + "\n")
        file.close()

def main():
    # Record the start time
    start_time = time.time()
    p = backpropagation()
    p.beforeTrainingModel()
    before_training = time.time()
    p.feedforwardTraining()
    print("The total training time",(time.time() - before_training)," secondes, which is ", (time.time() - before_training)/60, " minitues")
    p.defineConfusionMatrixForTestingData()
    p.defineConfusionMatrixForTrainingData()
    p.writeReport()
    print("The total time of training this model is ", (time.time() - start_time), "secondes which is ",(time.time() - start_time)/60," minitues." )

main()

