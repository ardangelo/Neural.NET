namespace Network
open Activations
open Costs
open MathNet.Numerics.LinearAlgebra

type Network(sizes : int list, activation, prime, weights : Matrix<double> list, biases : Vector<double> list) = 

    public new(activation, prime, weights : double list list list, biases : double list list) =
        let weightMatrices = weights |> List.map (fun weight -> DenseMatrix.ofRowList(weight))
        let biasVectors = biases |> List.map (fun bias -> DenseVector.ofList(bias))
        let sizes = weights |> List.map (fun weight -> weight.Length)
        Network(sizes, activation, prime, weightMatrices, biasVectors)

    public new(sizes : int list) = 
        let weightMatrices = [for i in 0 .. sizes.Length - 2 do yield DenseMatrix.init sizes.[i + 1] sizes.[i] (fun r c -> System.Random().NextDouble())]
        let biasVectors = [for i in 0 .. sizes.Length - 2 do yield DenseVector.init sizes.[i + 1] (fun r -> System.Random().NextDouble())]
        Network(sizes, Activations.Sigmoid.Activation, Activations.Sigmoid.Prime, weightMatrices, biasVectors)

    member private this.activation = activation
    member private this.prime = prime

    member private this.weights : Matrix<double> list = weights
    member private this.biases : Vector<double> list = biases

    static member private WeightedInput(a : Vector<double>, w : Matrix<double>, b : Vector<double>) =
        (w * a) + b

    member this.FeedForward(a : Vector<double>) =
        Network.FeedForward(this.activation, ([a]), this.weights, this.biases).Head.Map(System.Func<double,double> this.activation)

    member this.FeedForward(input : double list) =
        let a = DenseVector.ofList(input)
        this.FeedForward(a)

    static member FeedForward(activation, z : Vector<double> list, w : Matrix<double> list, b : Vector<double> list) : Vector<double> list =
        if w.IsEmpty then z else
        
        let z' = Network.WeightedInput(z.Head.Map(System.Func<double,double> activation), w.Head, b.Head)

        Network.FeedForward(activation, List.Cons(z', z), w.Tail, b.Tail)