namespace MNIST.Layers
{
    [Serializable]
    public static class Flatten
    {
        public static float[] Forward(float[,,] input)
        {
            int c = input.GetLength(0);
            int h = input.GetLength(1);
            int w = input.GetLength(2);
            float[] flat = new float[c * h * w];
            int idx = 0;
            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        flat[idx++] = input[i, y, x];
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
