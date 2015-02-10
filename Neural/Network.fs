namespace Network
open Activations
open MathNet.Numerics.LinearAlgebra

type Network(sizes : int list, activation, prime, weights : Matrix<double> list, biases : Matrix<double> list) = 

    public new(activation, prime, weights : double list list list, biases : double list list list) =
        let weightMatrices = weights |> List.map (fun weight -> DenseMatrix.ofRowList(weight))
        let biasMatrices = biases |> List.map (fun bias -> DenseMatrix.ofRowList(bias))
        let sizes = weights |> List.map (fun weight -> weight.Length)
        Network(sizes, activation, prime, weightMatrices, biasMatrices)

    public new(sizes : int list) = 
        let weightMatrices = [for i in 0 .. sizes.Length - 2 do yield DenseMatrix.init sizes.[i + 1] sizes.[i] (fun r c -> System.Random().NextDouble())]
        let biasMatrices = [for i in 0 .. sizes.Length - 2 do yield DenseMatrix.init sizes.[i + 1] 1 (fun r c -> System.Random().NextDouble())]
        Network(sizes, Activations.Sigmoid.Activation, Activations.Sigmoid.Prime, weightMatrices, biasMatrices)

    member private this.activation = activation
    member private this.prime = prime

    member private this.weights : Matrix<double> list = weights
    member private this.biases : Matrix<double> list = biases

    static member private WeightedInput(a : Matrix<double>, w : Matrix<double>, b : Matrix<double>) =
        (w * a) + b

    member this.FeedForward(a : Matrix<double>) =
        if a.ColumnCount > 1 then
            raise (System.ArgumentException("Input matrix must have only a single column"))
        else 
            Network.FeedForward(this.activation, ([a]), this.weights, this.biases).Head.Map(System.Func<double,double> this.activation)

    member this.FeedForward(input : double list) =
        let a = DenseMatrix.ofColumnList([input])
        this.FeedForward(a)

    static member FeedForward(activation, z : Matrix<double> list, w : Matrix<double> list, b : Matrix<double> list) : Matrix<double> list =
        if w.IsEmpty then z else
        
        let z' = Network.WeightedInput(z.Head.Map(System.Func<double,double> activation), w.Head, b.Head)

        Network.FeedForward(activation, List.Cons(z', z), w.Tail, b.Tail)