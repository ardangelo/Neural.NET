namespace Network
open MathNet.Numerics.LinearAlgebra

module Output =

    let WeightedInput(a : Vector<double>, w : Matrix<double>, b : Vector<double>) =
        (w * a) + b

    let rec FeedForward(activation, z : Vector<double> list, w : Matrix<double> list, b : Vector<double> list) : Vector<double> list =
        if w.IsEmpty then z else
        
        let z' = WeightedInput(z.Head.Map activation, w.Head, b.Head)

        FeedForward(activation, List.Cons(z', z), w.Tail, b.Tail)