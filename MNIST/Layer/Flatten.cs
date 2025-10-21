namespace MNIST.Layers
{
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
    }
}
