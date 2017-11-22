import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate, y2indicator

class LRModule(object):
    def __init__(self):
        pass
    

    #epochs = 10000
    def train(self, X, Y, step_size=10e-7, epochs=10000):
        # Validation data set extracted from the training data
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
      #  X, Y = X[:-1000], Y[:-1000]
        X, Y = X[:-9000], Y[:-9000]
        N, D = X.shape
        Y.shape = [np.size(Y), 1]
        Yvalid.shape = [np.size(Yvalid), 1]
        
        
        #HW3_2. Initialize the weights W to small random numbers (variance - zero); 
        #       also initialize the bias b to zero
        #Add code here...

        self.W = np.random.normal(0,0,(D,1))
        self.b = 0



        #HW3_3. For the given number of epochs set, learn the weights 
        costs = []
        best_validation_error = 1
        for i in range(epochs):
                #HW3_4. Do forward propagation to calculate P(Y|X)
                #Add code here....

                pY = self.forward(X)

                #HW3_5. Perform gradient descent using equations (1) and (2) from the HW text
                #Add code here....

                self.W -= step_size * np.dot(np.transpose(X), np.subtract(pY, Y))
                self.b -= step_size * sum(np.subtract(pY, Y))

                #HW3_6. Using the validation data, compute P(Y|X_valid) using the forward algo
                #       Compute the sigmoid costs and append to array costs
                #       Check to set best_validation_error 
                #Add code here....

                pYValid = self.forward(Xvalid)
                sigmoidCost = sigmoid_cost(Yvalid, pYValid)
                costs.append(sigmoidCost)
#                print (i, " ", sigmoidCost)

                # errorRate = error_rate(Yvalid, pYValid)
                # if errorRate < best_validation_error:
                #     best_validation_error = errorRate

                errorRate = error_rate(Yvalid, np.argmax(pYValid, axis=1))
                #print("i:", i, "cost:", sigmoidCost, "error:", error_rate())
                if error_rate() < best_validation_error:
                    best_validation_error = error_rate()

        
        #HW3_7. Include the value for the best validation error in your final report
        #Add code here....
        print("Best validation error is ", best_validation_error)
        
        #HW3_8. Display the graph of the validation cost in your final report
        #Add code here....

        plt.plot(costs)
        plt.show()
        
    
    #. Implement the forward algorithm
    def forward(self, X):
        #Add code here....
        return sigmoid(X.dot(self.W) + self.b)

    
    #. Implement the prediction algorithm, calling forward
    def predict(self, X):
        #Add code here....
        return self.forward(X)
    
    #. Implement a method to compute accuracy or classification rate
    def score(self, X, Y):
        #Add code here....
        pY = self.predict(X)
        return (1-error_rate(Y, pY))

    
def main():
    #HW3_1. Train a LR classifier on the fer13 training dataset 
    #Add code here....

    # X, Y = getBinaryfer13Data('fer3and4train.csv')
    # model = LRModule()
    # model.train(X, Y)
    
     #HW3_9. After your training errors are sufficiently low, 
     #       apply the classifier on the test set, 
     #       show the classification accuracy in your final report
     #Add code here....


    Xtest, Ytest = getBinaryfer13Data("fer3and4test.csv")
    testModel = LRModule()
    testModel.train(Xtest, Ytest)
    print(testModel.score(Xtest, Ytest))
    
    
if __name__ == '__main__':
    main()
        