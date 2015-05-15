# Neural.NET
A machine learning library for .NET written in F#

## Current version

Neural.NET is currently under heavy development. The library currently initializes simple feed-forward networks, updates weights and biases using gradient descent and trains stochastically. The network's weights and biases are stored in [Math.NET Numerics](http://numerics.mathdotnet.com/) matrices and vectors, respectively. The included client, NeuralTeach, parses handwritten training digit data from the included MNIST dataset and calls the Neural.NET network initialization and stochastic training and evaluation methods to train a network. 

### API

As the library is under heavy development, examine the included client NeuralAppTest for example usage.

### Examples

As Neural.NET targets the Portable Class Library, it is trivial to integrate into, say, a Windows Phone 8 app.
I have also written [NeuralAppTest](https://github.com/excelangue/NeuralAppTest), a simple implementation of such a Neural.NET client.
Users can draw a number in the square box and the app displays what it thinks is the number. The weights and biases of the network come from the output of NeuralTeach. 

![Drawing a number](http://andrew.uni.cx/assets/images/neural-net/neural-draw.jpg)

![The result](http://andrew.uni.cx/assets/images/neural-net/neural-result.jpg)

## In progress

I am now working on abstracting different steps of the network initialization, updating, and training to be able to swap in different training algorithms, update methods, and even different types of neural networks as needed.
For example, the digit-recognition network is a simple feed-forward network trained stochastically, but the second target network, the `C. Elegans` connectome, is a recurrent network trained with a genetic algorithm. My goal is to be able to train various types of networks (starting with the `C. Elegans` connectome) using the same library and interface with a minimum of friction.