using MNIST.Tensor;
using System;

namespace MNIST.Layers
{
    [Serializable]
    public class Conv2D
    {
        public int InChannels, OutChannels, KernelSize;
        public Tensor4D Weights; // [out, in, kH, kW]
        public float[] Bias;

        private Tensor3D? lastInput;
        private Tensor3D? lastOutput; // for ReLU derivative

        public Conv2D(int inC, int outC, int k)
        {
            InChannels = inC;
            OutChannels = outC;
            KernelSize = k;
            Weights = new Tensor4D(outC, inC, k, k);
            Bias = new float[outC];

            var rand = new Random();
            float limit = (float)Math.Sqrt(6.0 / (inC * k * k + outC * k * k));
            for (int o = 0; o < outC; o++)
                for (int i = 0; i < inC; i++)
                    for (int y = 0; y < k; y++)
                        for (int x = 0; x < k; x++)
                            Weights.Set(o, i, y, x, (float)(rand.NextDouble() * 2 - 1) * limit);
        }

        public Tensor3D Forward(Tensor3D input)
        {
            lastInput = input;
            int inH = input.Range1;
            int inW = input.Range2;
            int outH = inH - KernelSize + 1;
            int outW = inW - KernelSize + 1;
            Tensor3D output = new Tensor3D(OutChannels, outH, outW);

            for (int o = 0; o < OutChannels; o++)
                for (int y = 0; y < outH; y++)
                    for (int x = 0; x < outW; x++)
                    {
                        float sum = Bias[o];
                        for (int i = 0; i < InChannels; i++)
                            for (int ky = 0; ky < KernelSize; ky++)
                                for (int kx = 0; kx < KernelSize; kx++)
                                    sum += input.Get(i, y + ky, x + kx) * Weights.Get(o, i, ky, kx);
                        output.Set(o, y, x, Activation.ReLU(sum));
                    }

            lastOutput = output;
            return output;
        }

        public Tensor3D Backward(Tensor3D dOut, float lr)
        {
            int inH = lastInput!.Range1;
            int inW = lastInput.Range2;
            int outH = dOut.Range1;
            int outW = dOut.Range2;

            Tensor3D dInput = new Tensor3D(InChannels, inH, inW);

            for (int o = 0; o < OutChannels; o++)
            {
                for (int i = 0; i < InChannels; i++)
                {
                    float biosGrad = 0f;
                    for (int y = 0; y < outH; y++)
                    {
                        for (int x = 0; x < outW; x++)
                        {
                            float grad = dOut.Get(o, y, x);
                            if (lastOutput!.Get(o, y, x) <= 0) grad = 0;

                            for (int ky = 0; ky < KernelSize; ky++)
                            {
                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int iy = y + ky;
                                    int ix = x + kx;

                                    dInput.Set(i, iy, ix, dInput.Get(i, iy, ix) + grad * Weights.Get(o, i, ky, kx));

                                    Weights.Set(o, i, ky, kx, Weights.Get(o, i, ky, kx) - lr * grad * lastInput.Get(i, iy, ix));
                                }
                            }
                            biosGrad += grad;
                        }
                        Bias[o] -= lr * biosGrad;
                    }
                }
            }

            return dInput;
        }
    }
}
