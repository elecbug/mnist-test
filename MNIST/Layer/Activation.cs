using MNIST.Tensor;
using System;

namespace MNIST.Layers
{
    [Serializable]
    public static class Activation
    {
        public static float ReLU(float x) => Math.Max(0, x);

        public static Tensor1D Softmax(Tensor1D x)
        {
            double max = double.NegativeInfinity;
            for (int i = 0; i < x.Range0; i++) 
                if (x.Get(i) > max) 
                    max = x.Get(i);
            double sum = 0;
            double[] exps = new double[x.Range0];
            for (int i = 0; i < x.Range0; i++)
            {
                exps[i] = Math.Exp(x.Get(i) - max);
                sum += exps[i];
            }
            Tensor1D p = new Tensor1D(x.Range0);
            for (int i = 0; i < x.Range0; i++)
                p.Set(i, (float)(exps[i] / sum));
            return p;
        }

        // LeakyReLU and helpers
        public static float LeakyReLU(float x, float alpha = 0.01f) => x > 0 ? x : alpha * x;

        public static Tensor1D LeakyReLUArray(Tensor1D x, float alpha = 0.01f)
        {
            var r = new Tensor1D(x.Range0);
            for (int i = 0; i < x.Range0; i++)
                r.Set(i, x.Get(i) > 0 ? x.Get(i) : alpha * x.Get(i));
            return r;
        }

        // Derivative using ACTIVATED values (sign tells us region)
        public static Tensor1D DLeakyFromActivated(Tensor1D activated, float alpha = 0.01f)
        {
            var d = new Tensor1D(activated.Range0);
            for (int i = 0; i < activated.Range0; i++)
                d.Set(i, activated.Get(i) > 0 ? 1f : alpha);
            return d;
        }

        public static Tensor1D Mul(Tensor1D a, Tensor1D b)
        {
            var r = new Tensor1D(a.Range0);
            for (int i = 0; i < a.Range0; i++)
                r.Set(i, a.Get(i) * b.Get(i));
            return r;
        }
    }
}
