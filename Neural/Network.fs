namespace Network
open Activations
open Costs
open MathNet.Numerics.LinearAlgebra

type Network(sizes : int list, activation : double -> double, prime : double -> double, weights : Matrix<double> list, biases : Vector<double> list) = 

    public new(activation, prime, weights : double list list list, biases : double list list) =
        let weightMatrices = weights |> List.map (fun weight -> DenseMatrix.ofRowList(weight))
        let biasVectors = biases |> List.map (fun bias -> DenseVector.ofList(bias))
        let sizes = weights |> List.map (fun weight -> weight.Length)
        Network(sizes, activation, prime, weightMatrices, biasVectors)

    public new(sizes : int list) = 
        let weightMatrices = [for i in 0 .. sizes.Length - 2 do yield DenseMatrix.init sizes.[i + 1] sizes.[i] (fun r c -> System.Random().NextDouble())]
        let biasVectors = [for i in 0 .. sizes.Length - 2 do yield DenseVector.init sizes.[i + 1] (fun r -> System.Random().NextDouble())]
        Network(sizes, Activations.Sigmoid.Activation, Activations.Sigmoid.Prime, weightMatrices, biasVectors)
    
    //we are making these Func<double,double> because Math.Net's Map doesn't understand F# function types properly
    member private this.activation : System.Func<double,double> = (System.Func<double,double> activation)
    member private this.actPrime : System.Func<double,double> = (System.Func<double,double> prime)

    member private this.cost = Costs.Quadratic.Cost
    member private this.partialCost = Costs.Quadratic.PartialCost

    member private this.weights : Matrix<double> list = weights
    member private this.biases : Vector<double> list = biases

    static member private WeightedInput(a : Vector<double>, w : Matrix<double>, b : Vector<double>) =
        (w * a) + b

// Calculate output of system

    member this.Output(a : Vector<double>) =
        this.FeedForward(a).Head.Map(this.activation)

    member this.Output(input : double list) = 
        let a = DenseVector.ofList(input)
        Array.toList(this.Output(a).ToArray())

// FeedForward functions

    member private this.FeedForward(a : Vector<double>) : Vector<double> list =
        Network.FeedForward(this.activation, ([a]), this.weights, this.biases)

    // FeedForward algorithm
    static member private FeedForward(activation, z : Vector<double> list, w : Matrix<double> list, b : Vector<double> list) : Vector<double> list =
        if w.IsEmpty then z else
        
        let z' = Network.WeightedInput(z.Head.Map activation, w.Head, b.Head)

        Network.FeedForward(activation, List.Cons(z', z), w.Tail, b.Tail)

// Error functions

    // Class functions

    member private this.OutputError(a : Vector<double>, y : Vector<double>) =
        Network.OutputError(this.activation, this.actPrime, this.partialCost, this.FeedForward(a).Head, y)

    member private this.NetworkError(a : Vector<double>, y : Vector<double>) =
        Network.NetworkError(this.activation, this.actPrime, this.partialCost, a, y, this.weights, this.biases)

    // Pure functions

    // Final error calulation
    static member private OutputError(activation, actPrime, partialCost, lastZ : Vector<double>, y) =
        Vector.op_DotMultiply(partialCost(lastZ.Map activation, y), lastZ.Map(actPrime))

    // Error of layer before
    static member private PreviousError(actPrime, nextError : Vector<double>, z : Vector<double>, nextWeight : Matrix<double>) =
        let weightedError = Matrix.op_Multiply(nextWeight.Transpose(), nextError)
        Vector.op_DotMultiply(weightedError, z.Map(actPrime))

    // Error of entire network
    static member private NetworkError(activation, actPrime, partialCost, a, y, w : Matrix<double> list, b) : Vector<double> list =
        let reverseW = List.rev(w)
        let reverseZ = Network.FeedForward(activation, [a], w, b)

        let rec helper(prevErrors : Vector<double> list, reverseZ : Vector<double> list, reverseW : Matrix<double> list) : Vector<double> list = 
            if reverseZ.Length = 0 then
                prevErrors
            else
            let prevErrors' = List.Cons(Network.PreviousError(actPrime, prevErrors.Head, reverseZ.Head, reverseW.Head), prevErrors)
            helper(prevErrors', reverseZ.Tail, reverseW.Tail)

        helper([Network.OutputError(activation, actPrime, partialCost, reverseZ.Head, y)], reverseZ, reverseW)