namespace MNIST.Layer
{
    public class BatchNorm
    {
        private int size;
        private float[] gamma;
        private float[] beta;

        private float[] mean = { };
        private float[] variance = { };

        private float[] lastInput = { };
        private float[] normed = { };

        private float eps = 1e-5f;

        public BatchNorm(int size)
        {
            this.size = size;
            gamma = new float[size];
            beta = new float[size];
            for (int i = 0; i < size; i++)
            {
                gamma[i] = 1f;
                beta[i] = 0f;
            }
        }

        public float[] Forward(float[] x)
        {
            lastInput = x;
            mean = new float[size];
            variance = new float[size];
            normed = new float[size];

            float m = x.Length;

            for (int i = 0; i < size; i++)
                mean[i] = x[i];

            float[] outp = new float[size];
            for (int i = 0; i < size; i++)
            {
                normed[i] = (x[i] - mean[i]) / MathF.Sqrt(variance[i] + eps);
                outp[i] = gamma[i] * normed[i] + beta[i];
            }
            return outp;
        }

        public float[] Backward(float[] dOut, float lr = 0.001f)
        {
            float[] dx = new float[size];
            for (int i = 0; i < size; i++)
            {
                float dgamma = dOut[i] * normed[i];
                float dbeta = dOut[i];

                gamma[i] -= lr * dgamma;
                beta[i] -= lr * dbeta;
                dx[i] = dOut[i] * gamma[i];
            }
            return dx;
        }
    }
}
