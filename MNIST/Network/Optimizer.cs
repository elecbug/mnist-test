using MNIST.Tensor;

namespace MNIST.Network
{
    [Serializable]
    public class Optimizer
    {
        public class Adam
        {
            private Dictionary<Tensor2D, (Tensor2D, Tensor2D)> mW = new();
            private Dictionary<Tensor2D, (Tensor2D, Tensor2D)> vW = new();
            private Dictionary<Tensor1D, (Tensor1D, Tensor1D)> mB = new();
            private Dictionary<Tensor1D, (Tensor1D, Tensor1D)> vB = new();

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

            public void Step(ref Tensor2D W, ref Tensor1D B, Tensor2D dW, Tensor1D dB)
            {
                t++;
                if (!mW.ContainsKey(W))
                    mW[W] = (new Tensor2D(W.Range0, W.Range1), new Tensor2D(W.Range0, W.Range1));
                if (!mB.ContainsKey(B))
                    mB[B] = (new Tensor1D(B.Range0), new Tensor1D(B.Range0));

                var (mw, vw) = mW[W];
                var (mb, vb) = mB[B];

                for (int i = 0; i < W.Range0; i++)
                    for (int j = 0; j < W.Range1; j++)
                    {
                        mw.Set(i, j, beta1 * mw.Get(i, j) + (1 - beta1) * dW.Get(i, j));
                        vw.Set(i, j, beta2 * vw.Get(i, j) + (1 - beta2) * dW.Get(i, j) * dW.Get(i, j));

                        float mwc = mw.Get(i, j) / (1 - MathF.Pow(beta1, t));
                        float vwc = vw.Get(i, j) / (1 - MathF.Pow(beta2, t));
                        W.Set(i, j, W.Get(i, j) - lr * mwc / (MathF.Sqrt(vwc) + eps));
                    }

                for (int i = 0; i < B.Range0; i++)
                {
                    mb.Set(i, beta1 * mb.Get(i) + (1 - beta1) * dB.Get(i));
                    vb.Set(i, beta2 * vb.Get(i) + (1 - beta2) * dB.Get(i) * dB.Get(i));

                    float mbc = mb.Get(i) / (1 - MathF.Pow(beta1, t));
                    float vbc = vb.Get(i) / (1 - MathF.Pow(beta2, t));
                    B.Set(i, B.Get(i) - lr * mbc / (MathF.Sqrt(vbc) + eps));
                }

                mW[W] = (mw, vw);
                mB[B] = (mb, vb);
            }
        }
    }
}
