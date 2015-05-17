namespace NeuralNet

module public Activations =
    type ActivationRecord = { act: System.Func<double,double>; prime: System.Func<double,double> }
    
    let Step = {
        act =
            let actSharp z =
                if z > 0.0 then 1.0 else 0.0
            (System.Func<double,double> actSharp); // gross, but the alternative is tons of casting
        prime =
            let primeSharp z = 0.0
            (System.Func<double,double> primeSharp)}

    let Sigmoid = {
        act =
            let actSharp z =
                1.0 / (1.0 + MathNet.Numerics.Constants.E ** (-1.0 * z))
            (System.Func<double,double> actSharp);
        prime =
            let primeSharp z =
                let act = 1.0 / (1.0 + MathNet.Numerics.Constants.E ** (-1.0 * z))
                act * (1.0 - act)
            (System.Func<double,double> primeSharp)}