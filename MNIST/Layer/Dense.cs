namespace MNIST.Layers
{
    public class Dense
    {
        private float[,] Weights;
        private float[] Bias;
        private float[] lastInput;
        private float[] lastOutput;

        public Dense(int inputSize, int outputSize)
        {
            Weights = new float[outputSize, inputSize];
            Bias = new float[outputSize];
            var rand = new Random();
            for (int i = 0; i < outputSize; i++)
                for (int j = 0; j < inputSize; j++)
                    Weights[i, j] = (float)(rand.NextDouble() - 0.5) * 0.1f;
        }

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
                output[o] = Activation.ReLU(sum);
            }
            lastOutput = output;
            return output;
        }

        public float[] Backward(float[] dOut, float lr)
        {
            int inSize = Weights.GetLength(1);
            int outSize = Weights.GetLength(0);
            float[] dInput = new float[inSize];

            for (int o = 0; o < outSize; o++)
            {
                float grad = dOut[o] * (lastOutput[o] > 0 ? 1 : 0); // ReLU 미분
                for (int i = 0; i < inSize; i++)
                {
                    dInput[i] += grad * Weights[o, i];
                    Weights[o, i] -= lr * grad * lastInput[i];
                }
                Bias[o] -= lr * grad;
            }
            return dInput;
        }
    }
}
