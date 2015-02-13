// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
open NeuralNet

[<EntryPoint>]
let main argv = 
    let weights : double list list list = [
        [[1.0;0.0]; [-2.0;-2.0]; [0.0;1.0]] 
        [[-2.0;-2.0;0.0]; [0.0;-2.0;-2.0]; [0.0;1.0;0.0]]
        [[-2.0;-2.0;0.0]; [0.0;0.0;-4.0]]]
    let biases : double list list = [
        [0.0; 3.0; 0.0]
        [3.0; 3.0; 0.0]
        [3.0; 3.0]]
        
    let input : double list = argv |> Array.map double |> Seq.toList

    let network = Network.OfLists(Activations.Step.Activation, Activations.Step.Prime, weights, biases)
    let output = network.Output input

    let sum = int(output.[0])
    let carry = int(output.[1])

    System.Console.WriteLine(printfn "Sum: %i Carry: %i" sum carry)
    0 // return an integer exit code
    