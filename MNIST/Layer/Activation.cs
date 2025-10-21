namespace MNIST.Layers
{
    public static class Activation
    {
        public static float ReLU(float x) => Math.Max(0, x);

        public static float[] Softmax(float[] x)
        {
            float max = x.Max();
            float sum = 0;
            float[] exp = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
                sum += (exp[i] = (float)Math.Exp(x[i] - max));
            for (int i = 0; i < x.Length; i++)
                exp[i] /= sum;
            return exp;
        }
    }
}
