using MNIST.Tensor;
using System;

namespace MNIST.Layers
{
    [Serializable]
    public class Dense
    {
        public Tensor2D Weights;
        public Tensor1D Bias;

        // Grad buffers (accumulated over a mini-batch)
        public Tensor2D dWeights;
        public Tensor1D dBias;

        // Cache for backprop
        private Tensor1D? lastInput;

        public Dense(int inputSize, int outputSize)
        {
            Weights = new Tensor2D(outputSize, inputSize);
            Bias = new Tensor1D(outputSize);
            dWeights = new Tensor2D(outputSize, inputSize);
            dBias = new Tensor1D(outputSize);

            var rand = new Random();
            // Xavier init for stability
            float limit = (float)Math.Sqrt(6.0 / (inputSize + outputSize));
            for (int o = 0; o < outputSize; o++)
                for (int i = 0; i < inputSize; i++)
                    Weights.Set(o, i, (float)(rand.NextDouble() * 2 - 1) * limit);
        }

        // Linear forward (no activation here)
        public Tensor1D Forward(Tensor1D input)
        {
            lastInput = input;
            int outSize = Weights.Range0;
            int inSize = Weights.Range1;
            Tensor1D output = new Tensor1D(outSize);

            for (int o = 0; o < outSize; o++)
            {
                float sum = Bias.Get(o);
                for (int i = 0; i < inSize; i++)
                    sum += input.Get(i) * Weights.Get(o, i);
                output.Set(o, sum); // no activation here
            }
            return output;
        }

        // dOut is the gradient wrt this layer's output (pre-activation)
        public Tensor1D Backward(Tensor1D dOut)
        {
            int outSize = Weights.Range0;
            int inSize = Weights.Range1;
            Tensor1D dInput = new Tensor1D(inSize);

            // Accumulate gradients (DON'T clear here; ZeroGrad does that)
            for (int o = 0; o < outSize; o++)
            {
                float g = dOut.Get(o);
                for (int i = 0; i < inSize; i++)
                {
                    dInput.Set(i, dInput.Get(i) + g * Weights.Get(o, i));
                    dWeights.Set(o, i, dWeights.Get(o, i) + g * lastInput!.Get(i));
                }
                dBias.Set(o, dBias.Get(o) + g);
            }
            return dInput;
        }
    }
}
