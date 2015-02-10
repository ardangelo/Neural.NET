// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
open Network

[<EntryPoint>]
let main argv = 
    let weights : double list list list = [
        [[1.0;0.0]; [-2.0;-2.0]; [0.0;1.0]] 
        [[-2.0;-2.0;0.0]; [0.0;-2.0;-2.0]; [0.0;-4.0;0.0]]
        [[-2.0;-2.0;0.0]; [0.0;0.0;1.0]]]
    let biases : double list list list = [
        [[0.0]; [3.0]; [0.0]]
        [[3.0]; [3.0]; [3.0]]
        [[3.0]; [0.0]]]
        
    let input : double list = argv |> Array.map double |> Seq.toList
    let network = Network(Activations.Step.Activation, Activations.Step.Prime, weights, biases)
    let outMatrix = network.FeedForward input
    let sum = int(outMatrix.At(0, 0))
    let carry = int(outMatrix.At(1, 0))

    System.Console.WriteLine(printfn "Sum: %i Carry: %i" sum carry)
    0 // return an integer exit code
    