namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module public Costs =
    type Quadratic =
        static member Cost (a : Vector<double>) (y : Vector<double>) : double =
            0.5 * (y - a).Norm(2.0) ** 2.0

        static member PartialCost actPrime (z : Vector<double>) (a : Vector<double>) (y : Vector<double>) : Vector<double> =
            Vector.op_DotMultiply((a - y), z.Map(actPrime, Zeros.Include))

    type CrossEntropy =
        static member Cost (a : Vector<double>) (y : Vector<double>) : double =
            let one = DenseVector.ofList([for i in 1 .. a.Count do yield 1.0])
            let res = Vector.op_DotMultiply(-1.0 * y, a.PointwiseLog()) - Vector.op_DotMultiply((one - y), (one - a).PointwiseLog())
            res.Sum()

        static member PartialCost actPrime (z : Vector<double>) (a : Vector<double>) (y : Vector<double>) : Vector<double> =
            (a - y)