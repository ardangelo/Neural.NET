namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Learn =
    //updateVector is the method that takes a batch and some abstract vector `v` and makes it more "fit" using the batch data
    //StochasticTraining could be swapped out with another training algorithm
    //basic *Training should have updateVector, v, examples, testData, agent
    let rec StochasticTraining updateVector v (examples : 'a list) (testData : 'a list) (agent : MailboxProcessor<string> option) batchSize epochs =
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

        let rec UpdateUsingBatches (batches : 'a list list) (w, b) =
            if batches.IsEmpty then (w, b) else
            UpdateUsingBatches batches.Tail (updateVector batches.Head v)

        if epochs = 0 then v else

        if (agent.IsSome) then agent.Value.Post(sprintf "%d epochs left" epochs)

        let batches = Split (Shuffle examples) batchSize
        let v' = UpdateUsingBatches batches v
        
        if (agent.IsSome) then agent.Value.Post(sprintf "epoch complete")

        //test adjusted weights / biases
        (*let successes = 
            testData |> List.map (fun (x, y) -> 
                let a : Vector<double> = (NeuralNet.Output.FeedForward activation [x] w' b').Head.Map(activation, Zeros.Include)
                if a.MaximumIndex() = y.MaximumIndex() then 1 else 0) |> List.sum
        if (agent.IsSome) then agent.Value.Post(sprintf "%d / %d correct" successes testData.Length) *)

        StochasticTraining updateVector v' examples testData agent batchSize (epochs - 1)