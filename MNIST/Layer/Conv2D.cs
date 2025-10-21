namespace MNIST.Layers
{
    public class Conv2D
    {
        public int InChannels, OutChannels, KernelSize;
        public float[,,,] Weights; // [out, in, kH, kW]
        public float[] Bias;

        private float[,,] lastInput; // store input for backprop

        public Conv2D(int inC, int outC, int k)
        {
            InChannels = inC;
            OutChannels = outC;
            KernelSize = k;
            Weights = new float[outC, inC, k, k];
            Bias = new float[outC];

            var rand = new Random();
            for (int o = 0; o < outC; o++)
                for (int i = 0; i < inC; i++)
                    for (int y = 0; y < k; y++)
                        for (int x = 0; x < k; x++)
                            Weights[o, i, y, x] = (float)(rand.NextDouble() - 0.5) * 0.1f;
        }

        public float[,,] Forward(float[,,] input)
        {
            lastInput = input;
            int inH = input.GetLength(1);
            int inW = input.GetLength(2);
            int outH = inH - KernelSize + 1;
            int outW = inW - KernelSize + 1;
            float[,,] output = new float[OutChannels, outH, outW];

            for (int o = 0; o < OutChannels; o++)
                for (int y = 0; y < outH; y++)
                    for (int x = 0; x < outW; x++)
                    {
                        float sum = Bias[o];
                        for (int i = 0; i < InChannels; i++)
                            for (int ky = 0; ky < KernelSize; ky++)
                                for (int kx = 0; kx < KernelSize; kx++)
                                    sum += input[i, y + ky, x + kx] * Weights[o, i, ky, kx];
                        output[o, y, x] = Activation.ReLU(sum);
                    }

            return output;
        }

        public float[,,] Backward(float[,,] dOut, float lr)
        {
            float[,,] dInput = new float[InChannels, lastInput.GetLength(1), lastInput.GetLength(2)];

            for (int o = 0; o < OutChannels; o++)
                for (int i = 0; i < InChannels; i++)
                    for (int y = 0; y < dOut.GetLength(1); y++)
                        for (int x = 0; x < dOut.GetLength(2); x++)
                        {
                            float grad = dOut[o, y, x];
                            for (int ky = 0; ky < KernelSize; ky++)
                                for (int kx = 0; kx < KernelSize; kx++)
                                {
                                    int iy = y + ky;
                                    int ix = x + kx;
                                    dInput[i, iy, ix] += grad * Weights[o, i, ky, kx];
                                    Weights[o, i, ky, kx] -= lr * grad * lastInput[i, iy, ix];
                                }
                            Bias[o] -= lr * grad;
                        }
            return dInput;
        }
    }
}
