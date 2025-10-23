namespace MNIST.Network
{
    [Serializable]
    public class Optimizer
    {
        public class Adam
        {
            private Dictionary<float[,], (float[,], float[,])> mW = new();
            private Dictionary<float[,], (float[,], float[,])> vW = new();
            private Dictionary<float[], (float[], float[])> mB = new();
            private Dictionary<float[], (float[], float[])> vB = new();

            private float beta1 = 0.9f;
            private float beta2 = 0.999f;
            private float eps = 1e-8f;
            private float lr;
            private int t = 0;

            public float LearningRate { get=>lr; set=>lr=value; }

            public Adam(float learningRate = 0.001f)
            {
                lr = learningRate;
            }

            public void Step(ref float[,] W, ref float[] B, float[,] dW, float[] dB)
            {
                t++;
                if (!mW.ContainsKey(W))
                    mW[W] = (new float[W.GetLength(0), W.GetLength(1)], new float[W.GetLength(0), W.GetLength(1)]);
                if (!mB.ContainsKey(B))
                    mB[B] = (new float[B.Length], new float[B.Length]);

                var (mw, vw) = mW[W];
                var (mb, vb) = mB[B];

                for (int i = 0; i < W.GetLength(0); i++)
                    for (int j = 0; j < W.GetLength(1); j++)
                    {
                        mw[i, j] = beta1 * mw[i, j] + (1 - beta1) * dW[i, j];
                        vw[i, j] = beta2 * vw[i, j] + (1 - beta2) * dW[i, j] * dW[i, j];

                        float mwc = mw[i, j] / (1 - MathF.Pow(beta1, t));
                        float vwc = vw[i, j] / (1 - MathF.Pow(beta2, t));
                        W[i, j] -= lr * mwc / (MathF.Sqrt(vwc) + eps);
                    }

                for (int i = 0; i < B.Length; i++)
                {
                    mb[i] = beta1 * mb[i] + (1 - beta1) * dB[i];
                    vb[i] = beta2 * vb[i] + (1 - beta2) * dB[i] * dB[i];

                    float mbc = mb[i] / (1 - MathF.Pow(beta1, t));
                    float vbc = vb[i] / (1 - MathF.Pow(beta2, t));
                    B[i] -= lr * mbc / (MathF.Sqrt(vbc) + eps);
                }

                mW[W] = (mw, vw);
                mB[B] = (mb, vb);
            }
        }
    }
}
