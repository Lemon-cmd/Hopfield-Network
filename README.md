# Hopfield-Network
An implementation of the Hopfield network (Matrix/Eigen Based)

### What is Hopfield Network?
The Hopfield network is the first form of recurrent neural networks and it was developed by John Hopfield in the 1980s. 

### What separates this model from the modern ANNs?
Similar to other first variations of neural network, the Hopfield network is a single layer network. A distinct factor of this model is that it utilizes the ***Hebbian Rule of learning*** and of course, it is also a recurrent neural network. However, it is an ***unsupervised model*** and utilizes the McCullough-Pitt neurons which are sign neurons, whose activation function is clipping their values between (-1, 1) and 0 for unactive. 

### The Hebb's Rule 
Unlike current feed-forward phase of ANNs and their utilization of strictly ***restricted Boltzmann Networks***, the Hopfield model seeks to model after the operation of our biological neurons within our brain. This means only neurons who are active together, shall fire together. 

This rule is still very important as the modern ANNs architecture created by Geoffrey Hinton from Univ. Toronto does not follow it and thus, violate Dale's law.  
It remains essential as researchers are trying to develop a new way to not depend on backpropagation but rather, a process that updates during the feed forward phase. 

### Energy Function
Another distinction of this model is its usage of the energy function which serves as J or the cost function: 

***-(1/2) s.T * W * s ; s = state of each neurons, W = weights*** 

This equation is rewritten as -(1/2) * s * W * s.T since the outputs are vectors or matrixes of (1 X N) where N stands for num of neurons.
***If E = 0, then there is no change. If E < 0, then the model is performing well.***

### Training the Model 
The model is trained by storing patterns into the weight matrix where the number of patterns ***P*** is approximately equals to ***N / 2 * log2(N)*** where N is the number of neurons. Thus, this phase is the ***initilization of weights***

***W = X[0].T * X[0] + X[1].T * X[1] ... + X[i].T * X[i]***

### Spurious State
These states are often called spurious attractors. They simply denote a change in the states of the vector S.

Note: It is best to keep in mind that this model is ***binary*** and it can be trained unsupervisedly to recognize images of numbers (1 to 10).   
An example of the number 1. You can replace x with -1 to denote a blank in the image.          
    
[       
[x,x,x,1,x]     
[x,x,1,1,x]    
[x,1,x,1,x]    
[x,x,x,1,x]         
[x,1,1,1,1]        
]




