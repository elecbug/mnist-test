namespace MNIST.Layers
{
    public class MaxPool2D
    {
        private int kernel;
        private int stride;

        private float[,,] lastInput;

        public MaxPool2D(int kernel = 2, int stride = 2)
        {
            this.kernel = kernel;
            this.stride = stride;
        }

        public float[,,] Forward(float[,,] input)
        {
            lastInput = input;
            int c = input.GetLength(0);
            int inH = input.GetLength(1);
            int inW = input.GetLength(2);
            int outH = inH / stride;
            int outW = inW / stride;

            float[,,] output = new float[c, outH, outW];
            for (int ch = 0; ch < c; ch++)
            {
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        float max = float.MinValue;
                        for (int ky = 0; ky < kernel; ky++)
                        {
                            for (int kx = 0; kx < kernel; kx++)
                            {
                                int iy = y * stride + ky;
                                int ix = x * stride + kx;
                                if (iy < inH && ix < inW)
                                    max = Math.Max(max, input[ch, iy, ix]);
                            }
                        }
                        output[ch, y, x] = max;
                    }
                }
            }
            return output;
        }

        public float[,,] Backward(float[,,] dOut)
        {
            int c = lastInput.GetLength(0);
            int inH = lastInput.GetLength(1);
            int inW = lastInput.GetLength(2);
            float[,,] dInput = new float[c, inH, inW];

            int outH = dOut.GetLength(1);
            int outW = dOut.GetLength(2);

            for (int ch = 0; ch < c; ch++)
            {
                for (int y = 0; y < outH; y++)
                {
                    for (int x = 0; x < outW; x++)
                    {
                        float max = float.MinValue;
                        int maxY = -1, maxX = -1;
                        for (int ky = 0; ky < kernel; ky++)
                        {
                            for (int kx = 0; kx < kernel; kx++)
                            {
                                int iy = y * stride + ky;
                                int ix = x * stride + kx;
                                if (iy < inH && ix < inW)
                                {
                                    float val = lastInput[ch, iy, ix];
                                    if (val > max)
                                    {
                                        max = val;
                                        maxY = iy;
                                        maxX = ix;
                                    }
                                }
                            }
                        }
                        if (maxY >= 0 && maxX >= 0)
                            dInput[ch, maxY, maxX] = dOut[ch, y, x];
                    }
                }
            }
            return dInput;
        }
    }
}
