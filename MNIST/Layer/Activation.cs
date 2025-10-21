using System;

namespace MNIST.Layers
{
    [Serializable]
    public static class Activation
    {
        public static float ReLU(float x) => Math.Max(0, x);

        public static float[] Softmax(float[] x)
        {
            double max = double.NegativeInfinity;
            for (int i = 0; i < x.Length; i++) if (x[i] > max) max = x[i];
            double sum = 0;
            double[] exps = new double[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                exps[i] = Math.Exp(x[i] - max);
                sum += exps[i];
            }
            float[] p = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                p[i] = (float)(exps[i] / sum);
            return p;
        }

        // LeakyReLU and helpers
        public static float LeakyReLU(float x, float alpha = 0.01f) => x > 0 ? x : alpha * x;

        public static float[] LeakyReLUArray(float[] x, float alpha = 0.01f)
        {
            var r = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                r[i] = x[i] > 0 ? x[i] : alpha * x[i];
            return r;
        }

        // Derivative using ACTIVATED values (sign tells us region)
        public static float[] DLeakyFromActivated(float[] activated, float alpha = 0.01f)
        {
            var d = new float[activated.Length];
            for (int i = 0; i < activated.Length; i++)
                d[i] = activated[i] > 0 ? 1f : alpha;
            return d;
        }

        public static float[] Mul(float[] a, float[] b)
        {
            var r = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                r[i] = a[i] * b[i];
            return r;
        }
    }
}
