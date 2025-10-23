using MNIST.Layer;
using MNIST.Layers;
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
        private float[] lastH1Activated = { };

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

        public (float[] logits, float[,,] convOut) Forward(float[,,] input, bool training = true)
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


        public void Backward(float[] dLoss, float lr = 0.001f)
        {
            var dFc2 = fc2.Backward(dLoss);
            var dDrop = drop1.Backward(dFc2);

            var dAct = Activation.DLeakyFromActivated(lastH1Activated);
            for (int i = 0; i < dDrop.Length; i++)
                dDrop[i] *= dAct[i];

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
            Array.Clear(fc1.dWeights, 0, fc1.dWeights.Length);
            Array.Clear(fc1.dBias, 0, fc1.dBias.Length);
            Array.Clear(fc2.dWeights, 0, fc2.dWeights.Length);
            Array.Clear(fc2.dBias, 0, fc2.dBias.Length);
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

        private float[,] Scale(float[,] arr, float scale)
        {
            var r = new float[arr.GetLength(0), arr.GetLength(1)];
            for (int i = 0; i < arr.GetLength(0); i++)
                for (int j = 0; j < arr.GetLength(1); j++)
                    r[i, j] = arr[i, j] * scale;
            return r;
        }

        private float[] Scale(float[] arr, float scale)
        {
            var r = new float[arr.Length];
            for (int i = 0; i < arr.Length; i++) r[i] = arr[i] * scale;
            return r;
        }

        private float[,,] To3D(float[] flat, int c, int h, int w)
        {
            float[,,] x = new float[c, h, w];
            int idx = 0;
            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < w; z++)
                        x[i, y, z] = flat[idx++];
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
