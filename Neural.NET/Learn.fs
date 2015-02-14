namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module Learn =
    // calculate weights, biases adjustments for a single batch
    let CalculateGradient(activation, actPrime, partialCost, batch : (Vector<double> * Vector<double>) list, w : Matrix<double> list, b : Vector<double> list) =

        let singleExCalc((x, y), nablaW, nablaB) =
            let (deltaNablaW, deltaNablaB) = NeuralNet.Error.Backpropagate(activation, actPrime, partialCost, x, y, w, b)
            List.unzip (List.zip (List.zip nablaW nablaB) (List.zip deltaNablaW deltaNablaB) |> List.map(fun ((nW, nB), (dNW, dNB)) -> 
                ((nW + dNW), (nB + dNB))))

        let rec entireBatchCalc(batch : (Vector<double> * Vector<double>) list, nablaW, nablaB) = 
            if (batch.Length = 0) then
                (nablaW, nablaB)
            else
            let (nablaW', nablaB') = singleExCalc(batch.Head, nablaW, nablaB)
            entireBatchCalc(batch.Tail, nablaW', nablaB')

        let (nablaW, nablaB) = List.unzip (List.zip w b |> List.map(fun (wl, bl) ->
            (DenseMatrix.zero wl.RowCount wl.ColumnCount, DenseVector.zero bl.Count)))

        (nablaW, nablaB)

    // calculate adjusted weights, biases from single batch
    let DescendBatch(activation, actPrime, partialCost, eta : double, batch : (Vector<double> * Vector<double>) list, w : Matrix<double> list, b : Vector<double> list) =
        let m = batch.Length
        let (nablaW, nablaB) = CalculateGradient(activation, actPrime, partialCost, batch, w, b)

        let w' = List.zip w nablaW |> List.map(fun (wl, nwl) -> wl - ((eta/(double)m) * nwl))
        let b' = List.zip b nablaB |> List.map(fun (bl, nbl) -> bl - ((eta/(double)m) * nbl))

        (w', b')

    let Shuffle(target : 'a list) =
        let rnd = System.Random()
        let rec helper(target : 'a list, indices : int array) =
            if indices.Length = 0 then List.empty else
            let j = rnd.Next(0, indices.Length - 1)
            List.Cons(target.[indices.[j]], helper(target, indices |> Array.filter((<>)indices.[j])))
        helper(target, [|0 .. target.Length - 1|])

    let Split(target : 'a list, n : int) =
        let rec SplitAt(head : 'a list, tail : 'a list, n) =
            if n = 0 || tail.IsEmpty then (head, tail) else
            SplitAt(List.append head [tail.Head], tail.Tail, n - 1)
        let rec helper(target : 'a list, res : 'a list list) =
            if (target.Length <= n) then List.append res [target] else
            let (head, tail) = SplitAt([], target, n)
            helper(tail, List.append res [head])
        helper(target, [])

    // split up large amount of training examples and descend for each one, returning completed weights, biases
    // groups of N examples, all examples run through epochs times
    let rec StochasticGradientDescent(activation, actPrime, partialCost, eta, epochs, batchSize, examples : (Vector<double> * Vector<double>) list, w, b) =
        if epochs = 0 then (w, b) else

        let batches = Split(Shuffle(examples), batchSize)
        let rec DescendAllBatches(batches : (Vector<double> * Vector<double>) list list, w, b) =
            if batches.Length = 0 then (w, b) else
            let (w', b') = DescendBatch(activation, actPrime, partialCost, eta, batches.Head, w, b)
            DescendAllBatches(batches.Tail, w', b')

        let (w', b') = DescendAllBatches(batches, w, b)

        StochasticGradientDescent(activation, actPrime, partialCost, eta, epochs - 1, batchSize, examples, w', b')