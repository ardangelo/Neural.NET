namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

open Activations
open Costs
open Error

[<AutoOpen>]
type Network(activation : System.Func<double,double>, prime : System.Func<double,double>, weights : Matrix<double> list, biases : Vector<double> list) = 

    public new(activation, prime, weights : System.Collections.Generic.List<Matrix<double>>, biases : System.Collections.Generic.List<Vector<double>>) =
        Network(activation, prime, List.ofSeq weights, List.ofSeq biases)

    static member OfLists(activation, prime, weights : double list list list, biases : double list list) =
        let weightMatrices = weights |> List.map (fun weight -> DenseMatrix.ofRowList(weight))
        let biasVectors = biases |> List.map (fun bias -> DenseVector.ofList(bias))
        let sizes = weights |> List.map (fun weight -> weight.Length)
        Network(activation, prime, weightMatrices, biasVectors)

    static member Randomize(sizes : int list) = 
        let weightMatrices = [for i in 0 .. sizes.Length - 2 do yield DenseMatrix.init sizes.[i + 1] sizes.[i] (fun r c -> System.Random().NextDouble())]
        let biasVectors = [for i in 0 .. sizes.Length - 2 do yield DenseVector.init sizes.[i + 1] (fun r -> System.Random().NextDouble())]
        Network(Activations.Sigmoid.Activation, Activations.Sigmoid.Prime, weightMatrices, biasVectors)
    
    //we are making these Func<double,double> because Math.Net's Map doesn't understand F# function types properly
    member private this.activation = activation
    member private this.actPrime = prime

    member private this.cost = Costs.Quadratic.Cost
    member private this.partialCost = Costs.Quadratic.PartialCost

    member val private weights : Matrix<double> list = weights with get, set
    member val private biases : Vector<double> list = biases with get, set

// Calculate output of system

    member this.Output(a : Vector<double>) =
        this.FeedForward(a).Head.Map(this.activation)

    member this.Output(input : double list) = 
        let a = DenseVector.ofList(input)
        Array.toList(this.Output(a).ToArray())

// FeedForward function

    member private this.FeedForward(a : Vector<double>) : Vector<double> list =
        NeuralNet.Output.FeedForward(this.activation, ([a]), this.weights, this.biases)

// Error functions

    member private this.OutputError(a : Vector<double>, y : Vector<double>) =
        NeuralNet.Error.OutputError(this.activation, this.actPrime, this.partialCost, this.FeedForward(a).Head, y)

    member private this.NetworkError(a : Vector<double>, y : Vector<double>) =
        let reverseZ = NeuralNet.Output.FeedForward(this.activation, [a], this.weights, this.biases)
        NeuralNet.Error.NetworkError(this.activation, this.actPrime, this.partialCost, reverseZ, y, this.weights)

// Learning function
    
    member private this.Teach(examples : (Vector<double> * Vector<double>) list) =
        let eta = 3.0
        let batchSize = 10
        let epochs = 30

        let (w', b') = Learn.StochasticGradientDescent(this.activation, this.actPrime, this.partialCost, eta, epochs, batchSize, examples, this.weights, this.biases)

        this.weights <- w'
        this.biases <- b'

        (w', b')