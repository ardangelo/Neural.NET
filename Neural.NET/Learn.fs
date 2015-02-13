namespace NeuralNet
open MathNet.Numerics.LinearAlgebra

module Learn =
    let CalculateGradient(activation, actPrime, partialCost, batch : (Vector<double> * Vector<double>) list, w : Matrix<double> list, b : Vector<double> list) =

        let singleExCalc((x, y), nablaW, nablaB) =
            let (deltaNablaW, deltaNablaB) = NeuralNet.Error.Backpropagate(activation, actPrime, partialCost, x, y, w, b)
            List.unzip (List.zip (List.zip nablaW nablaB) (List.zip deltaNablaW deltaNablaB) |> List.map(fun ((nW, nB), (dNW, dNB)) -> 
                ((nW + dNW), (nB + dNB))))

        let rec entireBatchCalc(batch : (Vector<double> * Vector<double>) list, nablaW, nablaB) = 
            if (batch.Length = 0) then
                (nablaW, nablaB)
            else
            let (nablaW', nablaB') = singleExCalc(batch.Head, nablaW, nablaB)
            entireBatchCalc(batch.Tail, nablaW', nablaB')

        let (nablaW, nablaB) = List.unzip (List.zip w b |> List.map(fun (wl, bl) ->
            (DenseMatrix.zero wl.RowCount wl.ColumnCount, DenseVector.zero bl.Count)))

        (nablaW, nablaB)

    let GradientDescent(activation, actPrime, partialCost, eta : double, m : int, batch, w : Matrix<double> list, b : Vector<double> list) =
        let nablaW = w |> List.map(fun wl -> DenseMatrix.zero wl.RowCount wl.ColumnCount)
        let nablaB = b |> List.map(fun bl -> DenseVector.zero bl.Count)

        let w' = List.zip w nablaW |> List.map(fun (wl, nwl) -> wl - ((eta/(double)m) * nwl))
        let b' = List.zip b nablaB |> List.map(fun (bl, nbl) -> bl - ((eta/(double)m) * nbl))

        (w', b')