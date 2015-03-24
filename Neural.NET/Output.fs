namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module private Output =

    let WeightedInput (a : Vector<double>) (w : Matrix<double>) (b : Vector<double>) =
        (w * a) + b

    let rec FeedForward activation (z : Vector<double> list) (w : Matrix<double> list) (b : Vector<double> list) : Vector<double> list =
        if w.IsEmpty then z else
        let a = z.Head.Map(activation, Zeros.Include)
        let z' = WeightedInput a w.Head b.Head

        FeedForward activation (z'::z) w.Tail b.Tail

    let SoftMax (lastZ : Vector<double>) =
        let denom = lastZ.PointwiseExp().Sum()
        lastZ.Map(fun zi -> (MathNet.Numerics.Constants.E ** zi) / denom)