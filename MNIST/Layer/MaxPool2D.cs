using MNIST.Tensor;

namespace MNIST.Layers
{
    [Serializable]
    public class MaxPool2D
    {
        private int kernel;
        private int stride;

        private Tensor3D? lastInput;

        public MaxPool2D(int kernel = 2, int stride = 2)
        {
            this.kernel = kernel;
            this.stride = stride;
        }

        public Tensor3D Forward(Tensor3D input)
        {
            lastInput = input;
            int c = input.Range0;
            int inH = input.Range1;
            int inW = input.Range2;
            int outH = inH / stride;
            int outW = inW / stride;

            Tensor3D output = new Tensor3D(c, outH, outW);
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
                                    max = Math.Max(max, input.Get(ch, iy, ix));
                            }
                        }
                        output.Set(ch, y, x, max);
                    }
                }
            }
            return output;
        }

        public Tensor3D Backward(Tensor3D dOut)
        {
            int c = lastInput!.Range0;
            int inH = lastInput.Range1;
            int inW = lastInput.Range2;
            Tensor3D dInput = new Tensor3D(c, inH, inW);

            int outH = dOut.Range1;
            int outW = dOut.Range2;

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
                                    float val = lastInput.Get(ch, iy, ix);
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
                            dInput.Set(ch, maxY, maxX, dOut.Get(ch, y, x));
                    }
                }
            }
            return dInput;
        }
    }
}
