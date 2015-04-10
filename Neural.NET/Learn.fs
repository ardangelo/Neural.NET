namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Learn =    
    let GradientDescent activation actPrime partialCost eta (batch : (Vector<double> * Vector<double>) list) (w, b) =
        // calculate weights, biases adjustments for a single batch
        let CalculateGradient (batch : (Vector<double> * Vector<double>) list) =

            let (nablaW, nablaB) = List.unzip (List.map2(fun (wl:Matrix<double>) (bl:Vector<double>) ->
                (DenseMatrix.ofRowList([for i in 0 .. wl.RowCount - 1 do yield List.init wl.ColumnCount (fun _ -> 0.0)]), DenseVector.ofList(List.init bl.Count (fun _ -> 0.0)))) w b)

            let rec entireBatchCalc (batch : (Vector<double> * Vector<double>) list) (nablaW : Matrix<double> list) (nablaB : Vector<double> list) = 
                if (batch.Length = 0) then (nablaW, nablaB) else
                let (x, y) = batch.Head

                let (deltaNablaW, deltaNablaB) = NeuralNet.Error.Backpropagate activation actPrime partialCost x y w b
                let nablaW' = [for (nw, dnw) in (List.zip nablaW deltaNablaW) do yield nw + dnw]
                let nablaB' = [for (nb, dnb) in (List.zip nablaB deltaNablaB) do yield nb + dnb]
                entireBatchCalc batch.Tail nablaW' nablaB'

            entireBatchCalc batch nablaW nablaB
                        
        let m = (double)(batch.Length)
        let (nablaW, nablaB) = CalculateGradient batch

        let w' = [for (wl, nwl) in (List.zip w nablaW) do yield (wl - ((eta/m) * nwl))]
        let b' = [for (bl, bwl) in (List.zip b nablaB) do yield (bl - ((eta/m) * bwl))]

        (w', b')
        
    // split up large amount of training examples and descend for each one, returning completed weights, biases
    // groups of N examples, all examples run through epochs times
    let Shuffle (target : 'a list) : 'a list =
        let rnd = System.Random()
        let rec helper (target : 'a list) (indices : int array) (shuffled : 'a list) =
            if indices.Length = 0 then shuffled else
            let j = rnd.Next(0, indices.Length - 1)
            helper target (indices |> Array.filter((<>)indices.[j])) (target.[indices.[j]]::shuffled)
        helper target [|0 .. target.Length - 1|] ([] : 'a list)

    let Split (target : 'a list) (n : int) =
        let rec SplitAt (head : 'a list) (tail : 'a list) n =
            if n = 0 || tail.IsEmpty then (head, tail) else
            SplitAt (List.append head [tail.Head]) tail.Tail (n - 1)
        let rec helper (target : 'a list) (res : 'a list list) =
            if (target.Length <= n) then List.append res [target] else
            let (head, tail) = SplitAt [] target n
            helper tail (List.append res [head])
        helper target []

    //updateVector is the method that takes a batch and some abstract vector `v` and makes it more "fit" using the batch data
    //StochasticTraining could be swapped out with eg. GeneticTraining or other training algorithm
    let rec StochasticTraining updateVector v epochs batchSize (examples : 'a list) (testData : 'a list) (agent : MailboxProcessor<string> option) =
        let rec UpdateUsingBatch (batches : 'a list list) (w, b) =
            if batches.IsEmpty then (w, b) else
            UpdateUsingBatch batches.Tail (updateVector batches.Head v)

        if epochs = 0 then v else

        if (agent.IsSome) then agent.Value.Post(sprintf "%d epochs left" epochs)

        let batches = Split (Shuffle examples) batchSize
        let v' = UpdateUsingBatch batches v
        
        if (agent.IsSome) then agent.Value.Post(sprintf "epoch complete")

        //test adjusted weights / biases
        (*let successes = 
            testData |> List.map (fun (x, y) -> 
                let a : Vector<double> = (NeuralNet.Output.FeedForward activation [x] w' b').Head.Map(activation, Zeros.Include)
                if a.MaximumIndex() = y.MaximumIndex() then 1 else 0) |> List.sum
        if (agent.IsSome) then agent.Value.Post(sprintf "%d / %d correct" successes testData.Length) *)

        StochasticTraining updateVector v' (epochs - 1) batchSize examples testData agent