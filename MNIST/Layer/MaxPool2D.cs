namespace MNIST.Layers
{
    public class MaxPool2D
    {
        private int Kernel, Stride;
        public MaxPool2D(int kernel = 2, int stride = 2)
        {
            Kernel = kernel;
            Stride = stride;
        }

        public float[,,] Forward(float[,,] input)
        {
            int c = input.GetLength(0);
            int inH = input.GetLength(1);
            int inW = input.GetLength(2);
            int outH = inH / Stride;
            int outW = inW / Stride;
            float[,,] output = new float[c, outH, outW];

            for (int ch = 0; ch < c; ch++)
                for (int y = 0; y < outH; y++)
                    for (int x = 0; x < outW; x++)
                    {
                        float max = float.MinValue;
                        for (int ky = 0; ky < Kernel; ky++)
                            for (int kx = 0; kx < Kernel; kx++)
                            {
                                int iy = y * Stride + ky;
                                int ix = x * Stride + kx;
                                if (iy < inH && ix < inW)
                                    max = Math.Max(max, input[ch, iy, ix]);
                            }
                        output[ch, y, x] = max;
                    }
            return output;
        }
    }
}
