import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from helper import y2indicator, getBinaryfer13Data, sigmoid, sigmoid_cost, error_rate, softmax, cost


class NNModule(object):
    def __init__(self):
        pass
    
    def train(self, X, Y, step_size=10e-7, epochs=10000):
        # Validation data set extracted from the training data
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-10000], Y[:-10000]
        K = len(set(Y))
        
        # Convert outputs of the NN to an indicator matrix
        Ytrain_ind = y2indicator(Y, K)
        Yvalid_ind = y2indicator(Yvalid, K)
        M, D = X.shape

       

        #HW3_2. Randomly initialize all the hidden weights W's and biases b's 
        #Add code here....

        self.W1 = np.random.randn(D, M) / np.sqrt(D + M)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K) / np.sqrt(M + K)
        self.b2 = np.zeros(K)
        
        #HW3_3. For the given number of epochs set, implement backpropagation to learn the
        #       weights and append computed costs in the 2 cost arrays
        train_costs = []
        valid_costs = []
        best_validation_error = 1
        for i in range(epochs):
            #HW3_4. Run forward propagation twice; once to calculate P(Ytrain|X) 
            #       and Ztrain (activations at hidden layer); second to calculate
            #       P(Yvalid|Xvalid) and Zvalid
            #Add code here....

            pY, Z = self.forward(X)

            #HW3_5. Now we do back propagation by first performing 
            #       gradient descent using equations (3) and (4) from the HW text
            #Add code here....

            pY_T = pY - Ytrain_ind
            self.W2 -= step_size * (Z.T.dot(pY_T))
            self.b2 -= step_size * sum(pY_T)

            #HW3_5b. Now we do back propagation
            #Add code here....

            partialDiff = pY_T.dot(self.W2.T) * (1 - Z * Z)  # tanh BACKPROPAGATION
            self.W1 -= step_size * (X.T.dot(partialDiff))
            # self.b1 -= step_size * (dZ.sum(axis=0))
            self.b1 -= step_size * sum(partialDiff)
  
            #HW3_6. Compute the training and validation errors using cross_entropy cost
            #       function; once on the training predictions and once on validation predictions
            #       append errors to appropriate error array 
            #Add code here....

            pYValid, ZValid = self.forward(Xvalid)
            valid_costs.append(self.cross_entropy(Yvalid, pYValid))
            train_costs.append(self.cross_entropy(Y, pY))

            # errorRate = error_rate(Yvalid, pYValid)
            # if (errorRate < best_validation_error):
            #     best_validation_error = errorRate

            errorRate = error_rate(Yvalid, np.argmax(pYValid, axis=1))
            if errorRate < best_validation_error:
                best_validation_error = errorRate

        print("Valid cost: ", valid_costs)
        print("Train cost: ", train_costs)
        print("Best validation error: ", best_validation_error)




            
        #HW3_7. Include the best validation error and training and validation classification 
        #       rates in your final report
        #Add code here....
        
        #HW3_8. Display the training and validation cost graphs in your final report
        #Add code here....

        plt.plot(train_costs)
        plt.show()

        plt.plot(valid_costs)
        plt.show()




    # Implement the forward algorithm
    def forward(self, X):
        #Add code here....
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z

    # Implement the prediction algorithm
    def predict(self, P_Y_given_X):
        #Add code here....
        pY, _ = self.forward(P_Y_given_X)
        return np.argmax(pY, axis=1)

    # Implement a method to compute accuracy or classification rate
    def classification_rate(self, Y, P):
        #Add code here....
        return (1 - error_rate(Y, P))


    def cross_entropy(self, T, pY):
        #Add code here....
        N = len(T)
        return -np.log(pY[np.arange(N), T]).mean()


def main():
    #HW3_1. Train a NN classifier on the fer13 training dataset 
    #Add code here....

    X, Y = getBinaryfer13Data("fer3and4train.csv")
    model = NNModule()
    model.train(X, Y)

    Xtest, Ytest = getBinaryfer13Data("fer3and4test.csv")


    tScore = model.predict(Xtest)
    print("Score: ", model.classification_rate(Ytest, tScore))


    
    #HW3_9. After your training errors are sufficiently low, 
    #       apply the classifier on the test set, 
    #       show the classification accuracy in your final report
    #Add code here....

    
if __name__ == '__main__':
    main()
   