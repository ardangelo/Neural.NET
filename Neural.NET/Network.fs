namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

open Activations
open Costs
open Error

[<AutoOpen>]
type NeuralNetwork(activation : System.Func<double,double>, prime : System.Func<double,double>, weights : Matrix<double> list, biases : Vector<double> list, agent : MailboxProcessor<string> option) = 
    public new(activation, prime, weights : System.Collections.Generic.List<Matrix<double>>, biases : System.Collections.Generic.List<Vector<double>>, agent : MailboxProcessor<string> option) =
        NeuralNetwork(activation, prime, List.ofSeq weights, List.ofSeq biases, agent)

    static member Randomize(sizes : int list, agent : MailboxProcessor<string> option) =
        let rnd = System.Random()
        let weightMatrices = 
            [for i in 0 .. sizes.Length - 2 do 
                yield Matrix.Build.Random(sizes.[i + 1], sizes.[i], rnd.Next())]
        let biasVectors = 
            [for i in 1 .. sizes.Length - 1 do 
                yield Vector.Build.Random(sizes.[i], rnd.Next())]
        NeuralNetwork(Activations.Sigmoid.Activation, Activations.Sigmoid.Prime, weightMatrices, biasVectors, agent)

    static member Randomize(sizes : System.Collections.Generic.List<int>, agent : MailboxProcessor<string> option) =
        NeuralNetwork.Randomize(List.ofSeq sizes, agent)
    
    //we are making these Func<double,double> because Math.Net's Map doesn't understand F# function types properly
    member private this.activation = activation
    member private this.actPrime = prime

    member private this.cost = Costs.CrossEntropy.Cost
    member private this.partialCost = Costs.CrossEntropy.PartialCost

    member val weights : Matrix<double> list = weights with get, set
    member val biases : Vector<double> list = biases with get, set

    member this.agent : MailboxProcessor<string> option = agent
    
// Calculate output of system

    member this.Output(a : Vector<double>) = 
        this.FeedForward(a).Head.Map(this.activation, Zeros.Include)

    member this.ProbabilityDistribution(a : Vector<double>) =
        this.FeedForward(a).Head |> NeuralNet.Output.SoftMax
        
    member this.NetworkError(a : Vector<double>, y : Vector<double>) =
        let reverseZ = NeuralNet.Output.FeedForward this.activation [a] this.weights this.biases
        NeuralNet.Error.NetworkError this.activation this.actPrime this.partialCost reverseZ y this.weights |> List.toSeq

// FeedForward function

    member private this.FeedForward(a : Vector<double>) : Vector<double> list =
        NeuralNet.Output.FeedForward this.activation ([a]) this.weights this.biases

// Learning function
    
    member this.Teach(examples : (Vector<double> * Vector<double>) list, eta, batchSize, epochs, ?testData : (Vector<double> * Vector<double>) list) =
        let testData = defaultArg testData []

        let updateMethod = (Learn.GradientDescent this.activation this.actPrime this.partialCost eta)
        let (w', b') = Learn.StochasticTraining updateMethod (this.weights, this.biases) examples testData this.agent epochs batchSize

        this.weights <- w'
        this.biases <- b'

        (this.weights, this.biases)
