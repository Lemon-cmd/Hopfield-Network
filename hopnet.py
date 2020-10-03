import numpy as np
from math import log2 

class rnn():
    def __init__(self, N, M):
        self.__N = N
        self.__M = M
        self.__p = round(N/(2*log2(N))) #num of maximum patterns
        self.__W = None
        self.__s0 = np.ones((1,M))
        self.__s1 = np.zeros_like(self.__s0)
        
    def energy(self):
        return -0.5 * np.dot(np.dot(self.__s0, self.__W), self.__s0.T)

    def init_weights(self, X):
        #storing patterns p is approx (n/2log base2(n))
        for i in range(len(X)):
            if i == 0:
                self.__W = X[i].T * X[i] 
            else:
                self.__W = self.__W + X[i].T * X[i]

        self.__W = self.__W - self.__N * np.identity(self.__M)
        print("Trained Weights: \n", self.__W)
        print("\nInitial Energy: ", self.energy().sum())

    def train(self, X, epoch=25):
        e = 0
        while ((np.array_equal(self.__s0, self.__s1) != True) and (e < epoch)):
            self.__s0 = np.heaviside(np.dot(self.__s0, self.__W), -1)
            e += 1

        print(self.__s0)
        print("\nEnd Energy: ", self.energy().sum())

def gen_data(N, M):
    X, Y = [], []
    
    for r in range(N):
        X.append(np.random.randint(low=-1, high=2, size=(1, M)))
        Y.append(np.random.randint(low=-1, high=2, size=(1, M)))
    
    return X, Y

def main():
    X, Y = gen_data(30, 80) #N denotes number of rows or training points and M denotes cols which represent the range of patterns 2^M

    hop = rnn(50, 80) #50 neurons, 80 features
    
    #the higher N is, the more patterns are stored; however, the model is limited as it is only 1 layer
    #a limit is reached when the energy level does not go below 0
    hop.init_weights(X)
    hop.train(X, 20)
main()