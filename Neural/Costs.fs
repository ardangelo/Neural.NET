namespace Costs
open MathNet.Numerics.LinearAlgebra

type Quadratic =
    static member Cost(y : Vector<double>, a : Vector<double>) : double =
        (y - a).Norm(2.0) ** 2.0