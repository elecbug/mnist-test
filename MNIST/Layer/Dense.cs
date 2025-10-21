using System;

namespace MNIST.Layers
{
    [Serializable]
    public class Dense
    {
        public float[,] Weights;
        public float[] Bias;

        // Grad buffers (accumulated over a mini-batch)
        public float[,] dWeights;
        public float[] dBias;

        // Cache for backprop
        private float[] lastInput;

        public Dense(int inputSize, int outputSize)
        {
            Weights = new float[outputSize, inputSize];
            Bias = new float[outputSize];
            dWeights = new float[outputSize, inputSize];
            dBias = new float[outputSize];

            var rand = new Random();
            // Xavier init for stability
            float limit = (float)Math.Sqrt(6.0 / (inputSize + outputSize));
            for (int o = 0; o < outputSize; o++)
                for (int i = 0; i < inputSize; i++)
                    Weights[o, i] = (float)(rand.NextDouble() * 2 - 1) * limit;
        }

        // Linear forward (no activation here)
        public float[] Forward(float[] input)
        {
            lastInput = input;
            int outSize = Weights.GetLength(0);
            int inSize = Weights.GetLength(1);
            float[] output = new float[outSize];

            for (int o = 0; o < outSize; o++)
            {
                float sum = Bias[o];
                for (int i = 0; i < inSize; i++)
                    sum += input[i] * Weights[o, i];
                output[o] = sum; // no activation here
            }
            return output;
        }

        // dOut is the gradient wrt this layer's output (pre-activation)
        public float[] Backward(float[] dOut)
        {
            int outSize = Weights.GetLength(0);
            int inSize = Weights.GetLength(1);
            float[] dInput = new float[inSize];

            // Accumulate gradients (DON'T clear here; ZeroGrad does that)
            for (int o = 0; o < outSize; o++)
            {
                float g = dOut[o];
                for (int i = 0; i < inSize; i++)
                {
                    dInput[i] += g * Weights[o, i];
                    dWeights[o, i] += g * lastInput[i];
                }
                dBias[o] += g;
            }
            return dInput;
        }
    }
}
