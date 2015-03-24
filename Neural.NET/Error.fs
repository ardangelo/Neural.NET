namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Error =

    // Error of entire network
    let NetworkError activation actPrime partialCost (reverseZ : Vector<double> list) y (w : Matrix<double> list) : Vector<double> list =

        // Final layer error calulation
        let outputError =
            let lastA = reverseZ.Head.Map(activation, Zeros.Include)
            partialCost actPrime reverseZ.Head lastA y
        
        // Error of layer before
        let PreviousError (nextError : Vector<double>) (z : Vector<double>) (nextWeight : Matrix<double>) =
            let weightedError = Matrix.op_Multiply(nextWeight.Transpose(), nextError)
            let a = z.Map(actPrime, Zeros.Include)
            Vector.op_DotMultiply(weightedError, a)

        let reverseW = List.rev(w)

        let rec stepBack (nextErrors : Vector<double> list) (reverseZ : Vector<double> list) (reverseW : Matrix<double> list) : Vector<double> list = 
            if reverseZ.Length = 1 then nextErrors else //don't care about "error" of input

            let delta = PreviousError nextErrors.Head reverseZ.Head reverseW.Head
            stepBack (delta::nextErrors) reverseZ.Tail reverseW.Tail

        stepBack [outputError] reverseZ.Tail (List.rev w)

    let Backpropagate activation actPrime partialCost a y (w : Matrix<double> list) b : Matrix<double> list * Vector<double> list =
        let reverseZ = NeuralNet.Output.FeedForward activation [a] w b
        let error = NetworkError activation actPrime partialCost reverseZ y w

        (*let nablaW = [for i in 0 .. error.Length - 1 do yield (error.[i].ToColumnMatrix() * Matrix.transpose(reverseZ.[(reverseZ.Length - 1) - i].Map(activation, Zeros.Include).ToColumnMatrix()))]*)
        let nablaW = (List.rev reverseZ) |> (error |> List.map2(fun (delta:Vector<double>) (zi:Vector<double>) -> 
            delta.ToColumnMatrix() * Matrix.transpose(zi.Map(activation, Zeros.Include).ToColumnMatrix())))

        (nablaW, error)
