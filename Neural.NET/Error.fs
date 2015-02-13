namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module Error =

    // Final layer error calulation
    let OutputError(activation, actPrime, partialCost, lastZ : Vector<double>, y) =
        Vector.op_DotMultiply(partialCost(lastZ.Map activation, y), lastZ.Map(actPrime))

    // Error of layer before
    let PreviousError(actPrime, nextError : Vector<double>, z : Vector<double>, nextWeight : Matrix<double>) =
        let weightedError = Matrix.op_Multiply(nextWeight.Transpose(), nextError)
        Vector.op_DotMultiply(weightedError, z.Map(actPrime))

    // Error of entire network
    let NetworkError(activation, actPrime, partialCost, reverseZ : Vector<double> list, y, w : Matrix<double> list) : Vector<double> list =
        let reverseW = List.rev(w)

        let rec stepBack(prevErrors : Vector<double> list, reverseZ : Vector<double> list, reverseW : Matrix<double> list) : Vector<double> list = 
            if reverseZ.Length = 0 then
                prevErrors
            else
            let prevErrors' = List.Cons(PreviousError(actPrime, prevErrors.Head, reverseZ.Head, reverseW.Head), prevErrors)
            stepBack(prevErrors', reverseZ.Tail, reverseW.Tail)

        stepBack([OutputError(activation, actPrime, partialCost, reverseZ.Head, y)], reverseZ, reverseW)

    let Backpropagate(activation, actPrime, partialCost, a, y, w : Matrix<double> list, b) : Matrix<double> list * Vector<double> list =
        let reverseZ = NeuralNet.Output.FeedForward(activation, [a], w, b)
        let error = NetworkError(activation, actPrime, partialCost, reverseZ, y, w)
        let z = List.rev(reverseZ)

        let nablaW = [for i in 0 .. error.Length do yield (error.[i].ToColumnMatrix() * Matrix.transpose(z.[i].Map(activation).ToColumnMatrix()))]

        (nablaW, error)
