namespace Network

module Error =
    open MathNet.Numerics.LinearAlgebra

    // Final error calulation
    let OutputError(activation, actPrime, partialCost, lastZ : Vector<double>, y) =
        Vector.op_DotMultiply(partialCost(lastZ.Map activation, y), lastZ.Map(actPrime))

    // Error of layer before
    let PreviousError(actPrime, nextError : Vector<double>, z : Vector<double>, nextWeight : Matrix<double>) =
        let weightedError = Matrix.op_Multiply(nextWeight.Transpose(), nextError)
        Vector.op_DotMultiply(weightedError, z.Map(actPrime))

    // Error of entire network
    let NetworkError(activation, actPrime, partialCost, a, y, w : Matrix<double> list, b) : Vector<double> list =
        let reverseW = List.rev(w)
        let reverseZ = Network.Output.FeedForward(activation, [a], w, b)

        let rec stepBack(prevErrors : Vector<double> list, reverseZ : Vector<double> list, reverseW : Matrix<double> list) : Vector<double> list = 
            if reverseZ.Length = 0 then
                prevErrors
            else
            let prevErrors' = List.Cons(PreviousError(actPrime, prevErrors.Head, reverseZ.Head, reverseW.Head), prevErrors)
            stepBack(prevErrors', reverseZ.Tail, reverseW.Tail)

        stepBack([OutputError(activation, actPrime, partialCost, reverseZ.Head, y)], reverseZ, reverseW)