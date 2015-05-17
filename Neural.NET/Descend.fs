namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Descend =
    let GradientDescent activation cost eta (batch : (Vector<double> * Vector<double>) list) (w, b) =
        // calculate weights, biases adjustments for a single batch
        let CalculateGradient (batch : (Vector<double> * Vector<double>) list) =

            let (nablaW, nablaB) = List.unzip (List.map2(fun (wl:Matrix<double>) (bl:Vector<double>) ->
                (DenseMatrix.ofRowList([for i in 0 .. wl.RowCount - 1 do yield List.init wl.ColumnCount (fun _ -> 0.0)]), DenseVector.ofList(List.init bl.Count (fun _ -> 0.0)))) w b)

            let rec entireBatchCalc (batch : (Vector<double> * Vector<double>) list) (nablaW : Matrix<double> list) (nablaB : Vector<double> list) = 
                if (batch.Length = 0) then (nablaW, nablaB) else
                let (x, y) = batch.Head

                let (deltaNablaW, deltaNablaB) = NeuralNet.Error.Backpropagate activation cost x y w b
                let nablaW' = [for (nw, dnw) in (List.zip nablaW deltaNablaW) do yield nw + dnw]
                let nablaB' = [for (nb, dnb) in (List.zip nablaB deltaNablaB) do yield nb + dnb]
                entireBatchCalc batch.Tail nablaW' nablaB'

            entireBatchCalc batch nablaW nablaB
                        
        let m = (double)(batch.Length)
        let (nablaW, nablaB) = CalculateGradient batch

        let w' = [for (wl, nwl) in (List.zip w nablaW) do yield (wl - ((eta/m) * nwl))]
        let b' = [for (bl, bwl) in (List.zip b nablaB) do yield (bl - ((eta/m) * bwl))]

        (w', b')

    //Actually, I think that GeneticAlgorithm is more like "GradientDescent" than "StochasticTraining"; 
    //ST provides the data, GD / GT do the actual updating of the vector. (Rename GT accordingly)
    //we should have two modules: a "vector update" module (GD / GT) and "train on dataset" module (ST)

    let rec GeneticAlgorithm fitness population offspring generations =
        //current train: crossover [mutation, harmonization]. Todo: abstract training out into premade function like "updateVector"
        //current select: roulette. Abstract out as above
        let RouletteSelection (fitness : 'a -> double) (population : 'a list) : 'a list =
            //population |> List.sortBy fitness
            let rnd = new System.Random()
            [for indv in population do 
                let fit = fitness indv
                if rnd.NextDouble() < fit || fit = 1.0 then yield indv]
            
        let OrderedCrossover (parent1 : 'a array) (parent2 : 'a array) =
            //if parent1.Length <> parent2.Length then failWith "both parents must have same length" else

            let rnd = new System.Random()
            let (lo, hi) = 
                let a, b = rnd.Next(parent1.Length), rnd.Next(parent1.Length)
                if a < b then (a, b) else (b, a)
            let cross (pa : 'a array) (pb : 'a array) =
                let rest = pb |> Array.filter (fun x -> not (pa.[lo .. hi] |> Array.exists (fun n -> n = x)))
                [|for i in [0 .. parent1.Length - 1] do
                    if i < lo then yield rest.[i] else 
                    if i > hi then yield rest.[i - hi] else yield pa.[i]|]
            ((cross parent1 parent2), (cross parent2 parent1))

        let Mutate (c : 'a array) =
            let rnd = new System.Random()
            let a, b = rnd.Next(c.Length), rnd.Next(c.Length)
            let t = c.[b]
            c.[b] <- c.[a]
            c.[a] <- t
            c

        if generations = 0 then population else

        let pop' = RouletteSelection fitness population
        let rec CrossAllPairs (parents : 'a array list) =
            if parents.Length < 2 then parents else
            let p1::p2::rest = parents
            let (c1, c2) = (OrderedCrossover p1 p2)
            c1::c2::(CrossAllPairs rest)
        let pop' = CrossAllPairs pop'
        pop' |> List.map (fun c -> 
           let rnd = new System.Random()
           if rnd.NextDouble() < 0.05 then Mutate c else c)