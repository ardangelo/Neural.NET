namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

open Activations
open Costs
open Error

[<AutoOpen>]
type FeedForwardNetwork(activation : Activations.ActivationRecord, weights : Matrix<double> list, biases : Vector<double> list, agent : MailboxProcessor<string> option) = 
    public new(activation, weights : System.Collections.Generic.List<Matrix<double>>, biases : System.Collections.Generic.List<Vector<double>>, agent : MailboxProcessor<string> option) =
        FeedForwardNetwork(activation, List.ofSeq weights, List.ofSeq biases, agent)

    static member Randomize(sizes : int list, agent : MailboxProcessor<string> option) =
        let rnd = System.Random()
        let weightMatrices = 
            [for i in 0 .. sizes.Length - 2 do 
                yield Matrix.Build.Random(sizes.[i + 1], sizes.[i], rnd.Next())]
        let biasVectors = 
            [for i in 1 .. sizes.Length - 1 do 
                yield Vector.Build.Random(sizes.[i], rnd.Next())]
        FeedForwardNetwork(Activations.Sigmoid, weightMatrices, biasVectors, agent)

    static member Randomize(sizes : System.Collections.Generic.List<int>, agent : MailboxProcessor<string> option) =
        FeedForwardNetwork.Randomize(List.ofSeq sizes, agent)
    
    member private this.activation = Activations.Step

    member private this.cost = Costs.CrossEntropy

    member val weights : Matrix<double> list = weights with get, set
    member val biases : Vector<double> list = biases with get, set

    member this.agent : MailboxProcessor<string> option = agent
    
// Calculate output of system

    member this.Output(a : Vector<double>) = 
        this.FeedForward(a).Head.Map(this.activation.act, Zeros.Include)

    member this.ProbabilityDistribution(a : Vector<double>) =
        this.FeedForward(a).Head |> NeuralNet.Output.SoftMax
        
    member this.NetworkError(a : Vector<double>, y : Vector<double>) =
        let reverseZ = NeuralNet.Output.FeedForward this.activation [a] this.weights this.biases
        NeuralNet.Error.NetworkError this.activation this.cost reverseZ y this.weights |> List.toSeq

// FeedForward function

    member private this.FeedForward(a : Vector<double>) : Vector<double> list =
        NeuralNet.Output.FeedForward this.activation ([a]) this.weights this.biases

// Learning function
    
    member this.Teach(examples : (Vector<double> * Vector<double>) list, eta, batchSize, epochs, ?testData : (Vector<double> * Vector<double>) list) =
        let testData = defaultArg testData []

        let updateMethod = (Descend.GradientDescent this.activation this.cost eta)
        let (w', b') = Learn.StochasticTraining updateMethod (this.weights, this.biases) examples testData this.agent epochs batchSize

        this.weights <- w'
        this.biases <- b'

        (this.weights, this.biases)
