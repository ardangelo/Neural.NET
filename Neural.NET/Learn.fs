namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Learn =
    // calculate weights, biases adjustments for a single batch
    let CalculateGradient(activation, actPrime, partialCost, batch : (Vector<double> * Vector<double>) list, w : Matrix<double> list, b : Vector<double> list) =

        let (nablaW, nablaB) = List.unzip (List.zip w b |> List.map(fun (wl, bl) ->
            (DenseMatrix.ofRowList([for i in 0 .. wl.RowCount - 1 do yield List.init wl.ColumnCount (fun _ -> 0.0)]), DenseVector.ofList(List.init bl.Count (fun _ -> 0.0)))))

        let rec entireBatchCalc(batch : (Vector<double> * Vector<double>) list, nablaW : Matrix<double> list, nablaB : Vector<double> list) = 
            if (batch.Length = 0) then (nablaW, nablaB) else
            let (x, y) = batch.Head

            let (deltaNablaW, deltaNablaB) = NeuralNet.Error.Backpropagate(activation, actPrime, partialCost, x, y, w, b)
            let nablaW' = [for (nw, dnw) in (List.zip nablaW deltaNablaW) do yield nw + dnw]
            let nablaB' = [for (nb, dnb) in (List.zip nablaB deltaNablaB) do yield nb + dnb]
            entireBatchCalc(batch.Tail, nablaW', nablaB')

        entireBatchCalc(batch, nablaW, nablaB)

    let Shuffle(target : 'a list) : 'a list =
        let rnd = System.Random()
        let rec helper(target : 'a list, indices : int array, shuffled : 'a list) =
            if indices.Length = 0 then shuffled else
            let j = rnd.Next(0, indices.Length - 1)
            helper(target, indices |> Array.filter((<>)indices.[j]), List.Cons(target.[indices.[j]], shuffled))
        helper(target, [|0 .. target.Length - 1|], ([] : 'a list))

    let Split(target : 'a list, n : int) =
        let rec SplitAt(head : 'a list, tail : 'a list, n) =
            if n = 0 || tail.IsEmpty then (head, tail) else
            SplitAt(List.append head [tail.Head], tail.Tail, n - 1)
        let rec helper(target : 'a list, res : 'a list list) =
            if (target.Length <= n) then List.append res [target] else
            let (head, tail) = SplitAt([], target, n)
            helper(tail, List.append res [head])
        helper(target, [])

    let rec DescendAllBatches(activation, actPrime, partialCost, eta, batches : (Vector<double> * Vector<double>) list list, w, b) =
        if batches.Length = 0 then (w, b) else
            
        let m = (double)(batches.Head.Length)
        let (nablaW, nablaB) = CalculateGradient(activation, actPrime, partialCost, batches.Head, w, b)

        let w' = [for (wl, nwl) in (List.zip w nablaW) do yield (wl - ((eta/m) * nwl))]
        let b' = [for (bl, bwl) in (List.zip b nablaB) do yield (bl - ((eta/m) * bwl))]

        DescendAllBatches(activation, actPrime, partialCost, eta, batches.Tail, w', b')

    // split up large amount of training examples and descend for each one, returning completed weights, biases
    // groups of N examples, all examples run through epochs times
    let rec StochasticGradientDescent(activation, actPrime, partialCost, eta, epochs, batchSize, examples : (Vector<double> * Vector<double>) list, w, b, testData : (Vector<double> * Vector<double>) list, agent : MailboxProcessor<string> option) =
        if epochs = 0 then (w, b) else

        if (agent.IsSome) then agent.Value.Post(sprintf "%d epochs left" epochs)

        let batches = Split(Shuffle(examples), batchSize)
        let (w', b') = DescendAllBatches(activation, actPrime, partialCost, eta, batches, w, b)
        
        if (agent.IsSome) then agent.Value.Post(sprintf "epoch complete")

        //test adjusted weights / biases
        let success = 
            [for test in testData do 
                let a : Vector<double> = NeuralNet.Output.FeedForward(activation, [fst test], w', b').Head.Map(activation, Zeros.Include)
                yield if a.MaximumIndex() = (snd test).MaximumIndex() then 1 else 0] |> List.sum
        if (agent.IsSome) then agent.Value.Post(sprintf "%d / %d correct" success testData.Length)

        StochasticGradientDescent(activation, actPrime, partialCost, eta, epochs - 1, batchSize, examples, w', b', testData, agent)
