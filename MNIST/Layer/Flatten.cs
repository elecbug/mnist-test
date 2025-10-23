using MNIST.Tensor;

namespace MNIST.Layers
{
    [Serializable]
    public static class Flatten
    {
        public static Tensor1D Forward(Tensor3D input)
        {
            int c = input.Range0;
            int h = input.Range1;
            int w = input.Range2;
            Tensor1D flat = new Tensor1D(c * h * w);
            int idx = 0;
            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        flat.Set(idx, input.Get(i, y, x));
                        idx++;
                    }
            return flat;
        }

        public static float[,,] Unflatten(float[] flat, int c, int h, int w)
        {
            if (flat.Length != c * h * w)
                throw new ArgumentException($"Unflatten size mismatch: flat={flat.Length}, expected={c * h * w}");

            float[,,] output = new float[c, h, w];
            int idx = 0;

            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        output[i, y, x] = flat[idx++];

            return output;
        }
    }
}
