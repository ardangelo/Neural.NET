// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.

open MathNet.Numerics.LinearAlgebra
open System.IO
open NeuralNet

[<EntryPoint>]
let main argv = 
    let sizes = [784;30;10]

    let resultVectors = [
        DenseVector.ofList [1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0];
        DenseVector.ofList [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 1.0]];
    let learnfiles = [
        "../../../teach-data/0-learn";
        "../../../teach-data/1-learn";
        "../../../teach-data/2-learn";
        "../../../teach-data/3-learn";
        "../../../teach-data/4-learn";
        "../../../teach-data/5-learn";
        "../../../teach-data/6-learn";
        "../../../teach-data/7-learn";
        "../../../teach-data/8-learn";
        "../../../teach-data/9-learn"]

    let testfiles = [
        "../../../teach-data/0-test";
        "../../../teach-data/1-test";
        "../../../teach-data/2-test";
        "../../../teach-data/3-test";
        "../../../teach-data/4-test";
        "../../../teach-data/5-test";
        "../../../teach-data/6-test";
        "../../../teach-data/7-test";
        "../../../teach-data/8-test";
        "../../../teach-data/9-test"]
    
    let rec buildExamples(rs : Vector<double> list, fs : string list) = 
        if rs.Length = 0 then List.empty else
        let reader = new BinaryReader(File.Open(fs.Head, FileMode.Open))

        let toDouble(b1 : int, b2 : int) : double =
            let hex : string = new string([(char)(b1);(char)(b2)] |> Array.ofList)
            (double)(System.Int32.Parse(hex, System.Globalization.NumberStyles.AllowHexSpecifier))

        let readNextMNIST(br : BinaryReader) =
            DenseVector.ofList [for i in 1 .. 784 do yield toDouble(reader.Read(),reader.Read())]
        let images = (int)(reader.BaseStream.Length) / (784 * 2)
        let digits = [for i in 1 .. images do yield (readNextMNIST(reader), rs.Head)]

        List.append digits (buildExamples (rs.Tail, fs.Tail))

    let examples = buildExamples(resultVectors, learnfiles)
    let testdata = buildExamples(resultVectors, testfiles)

    let network = Network.Randomize(sizes)
    let (w, b) = network.Teach(examples, 3.0, 10, 30, testdata)

    
    let mutable filename = "output.cs"
    if argv.Length = 1 then
        filename <- argv.[0]

    let bw = new BinaryWriter(File.Open("output.txt", FileMode.Create))
    bw.Write("double[,] biases = new double[,] {\n")
    
    let rec writeVectorString(vects : Vector<double> list) =
        if vects.Length = 0 then bw.Write("};\n") else
        let vs = vects.Head.ToVectorString(System.Int32.MaxValue, 1, "F3")
        let line = "{" + vs.Replace("\r\n", ",")
        if vects.Length = 1 then 
            bw.Write(line.Substring(0, line.Length - 2) + "}\n")
        else
            bw.Write(line.Substring(0, line.Length - 2) + "}, \n")
        writeVectorString(vects.Tail)

    writeVectorString(b)

    bw.Write("double[,,] weights = new double[,,] {\n")

    let rec writeMatrixString(m : Matrix<double> list) =
        if m.Length = 0 then bw.Write("};\n") else

        bw.Write("{\n")
        let listOfVects = [for row in m.Head.ToRowArrays() do yield DenseVector.ofArray(row)]
        writeVectorString(listOfVects)

        if m.Length = 1 then 
            bw.Write("}\n")
        else
            bw.Write("}, \n")
        writeMatrixString(m.Tail)

    writeMatrixString(w)
    
    0 // return an integer exit code
