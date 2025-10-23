using MNIST.Layer;
using MNIST.Layers;
using MNIST.Tensor;
using System;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using static MNIST.Network.Optimizer;

namespace MNIST.Network
{
    [Serializable]
    public class CNNModel
    {
        private Conv2D conv1;
        private Conv2D conv2;
        private MaxPool2D pool1;
        private MaxPool2D pool2;
        private Dense fc1;
        private Dense fc2;
        private Dropout drop1;

        // cache for activation derivative
        private Tensor1D? lastH1Activated;

        public CNNModel()
        {
            conv1 = new Conv2D(1, 8, 3);
            conv2 = new Conv2D(8, 16, 3);
            pool1 = new MaxPool2D(2, 2);
            pool2 = new MaxPool2D(2, 2);
            fc1 = new Dense(16 * 5 * 5, 128);
            fc2 = new Dense(128, 10);
            drop1 = new Dropout(0.3f);
        }

        public (Tensor1D logits, Tensor3D convOut) Forward(Tensor3D input, bool training = true)
        {
            var x = conv1.Forward(input);
            x = pool1.Forward(x);
            x = conv2.Forward(x);
            x = pool2.Forward(x);

            var flat = Flatten.Forward(x);
            var h1 = Activation.LeakyReLUArray(fc1.Forward(flat));
            lastH1Activated = h1;
            h1 = drop1.Forward(h1, training);

            var outp = fc2.Forward(h1);
            return (Activation.Softmax(outp), x);
        }


        public void Backward(Tensor1D dLoss, float lr = 0.001f)
        {
            var dFc2 = fc2.Backward(dLoss);
            var dDrop = drop1.Backward(dFc2);

            var dAct = Activation.DLeakyFromActivated(lastH1Activated!);
            for (int i = 0; i < dDrop.Range0; i++)
                dDrop.Set(i, dDrop.Get(i) * dAct.Get(i));

            var dFc1 = fc1.Backward(dDrop);
            var dConvIn = To3D(dFc1, 16, 5, 5);
            var dPool2 = pool2.Backward(dConvIn);
            var dConv2 = conv2.Backward(dPool2, lr);
            var dPool1 = pool1.Backward(dConv2);
            conv1.Backward(dPool1, lr);
        }

        public void ZeroGrad()
        {
            // DO NOT reassign arrays; just clear them.
            fc1.dWeights.ArrayClear();
            fc1.dBias.ArrayClear();
            fc2.dWeights.ArrayClear();
            fc2.dBias.ArrayClear();
        }

        // Adam step for FC layers; convs are updated inside their Backward via lr (SGD)
        public void Step(Adam opt, float scale = 1f)
        {
            // Optionally scale grads if you averaged elsewhere
            opt.Step(ref fc1.Weights, ref fc1.Bias,
                     Scale(fc1.dWeights, scale), Scale(fc1.dBias, scale));
            opt.Step(ref fc2.Weights, ref fc2.Bias,
                     Scale(fc2.dWeights, scale), Scale(fc2.dBias, scale));
        }

        private Tensor2D Scale(Tensor2D arr, float scale)
        {
            var r = new Tensor2D(arr.Range0, arr.Range1);
            for (int i = 0; i < arr.Range0; i++)
                for (int j = 0; j < arr.Range1; j++)
                    r.Set(i, j, arr.Get(i, j) * scale);
            return r;
        }

        private Tensor1D Scale(Tensor1D arr, float scale)
        {
            var r = new Tensor1D(arr.Range0);
            for (int i = 0; i < arr.Range0; i++) 
                r.Set(i, arr.Get(i) * scale);
            return r;
        }

        private Tensor3D To3D(Tensor1D flat, int c, int h, int w)
        {
            Tensor3D x = new Tensor3D(c, h, w);
            int idx = 0;
            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < w; z++)
                    {
                        x.Set(i, y, z, flat.Get(idx));
                        idx++;
                    }
            return x;
        }

        public void Save(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
#pragma warning disable SYSLIB0011
            var formatter = new BinaryFormatter();
            formatter.Serialize(fs, this);
#pragma warning restore SYSLIB0011
        }

        public static CNNModel Load(string path)
        {
            using var fs = new FileStream(path, FileMode.Open);
#pragma warning disable SYSLIB0011
            var formatter = new BinaryFormatter();
            var model = (CNNModel)formatter.Deserialize(fs);
#pragma warning restore SYSLIB0011
            return model;
        }
    }
}
