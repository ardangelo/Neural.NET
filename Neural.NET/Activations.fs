namespace NeuralNet

module public Activations =
    type Step =
        static member Activation : System.Func<double,double> =
            let actSharp (z : double) : double =
                if z > 0.0 then 1.0 else 0.0
            (System.Func<double,double> actSharp) // gross, but the alternative is tons of casting

        static member Prime : System.Func<double,double> = 
            let primeSharp (z : double) : double = 0.0
            (System.Func<double,double> primeSharp)

    type Sigmoid =
        static member Activation : System.Func<double,double> =
            let actSharp (z : double) : double =
                1.0 / (1.0 + MathNet.Numerics.Constants.E ** (-1.0 * z))
            (System.Func<double,double> actSharp)

        static member Prime : System.Func<double,double> = 
            let primeSharp (z : double) : double =
                let act = 1.0 / (1.0 + MathNet.Numerics.Constants.E ** (-1.0 * z))
                act * (1.0 - act)
            (System.Func<double,double> primeSharp)