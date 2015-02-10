namespace Activations

type Step =
    static member Activation(z : double) : double =
        if z > 0.0 then 1.0 else 0.0
    static member Prime(z : double) : double = 0.0

type Sigmoid =
    static member Activation(z : double) : double =
        1.0 / (1.0 + MathNet.Numerics.Constants.E ** (-1.0 * z))
    static member Prime(z : double) : double =
        Sigmoid.Activation(z) * (1.0 - Sigmoid.Activation(z))