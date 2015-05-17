namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module public Costs =
    type CostRecord = { 
        cost: Vector<double> -> Vector<double> -> double; 
        partial: Activations.ActivationRecord -> Vector<double> -> Vector<double> -> Vector<double> -> Vector<double>}

    let Quadratic = {
        cost = fun a y ->
            0.5 * (y - a).Norm(2.0) ** 2.0;
        partial = fun activation z a y ->
            Vector.op_DotMultiply((a - y), z.Map(activation.prime, Zeros.Include))}

    let CrossEntropy = {
        cost = fun a y ->
            let one = DenseVector.ofList([for i in 1 .. a.Count do yield 1.0])
            let res = Vector.op_DotMultiply(-1.0 * y, a.PointwiseLog()) - Vector.op_DotMultiply((one - y), (one - a).PointwiseLog())
            res.Sum();
        partial = fun activation z a y ->
            (a - y)}