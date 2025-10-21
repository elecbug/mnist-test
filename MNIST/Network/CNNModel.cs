using MNIST.Layers;

namespace MNIST.Network
{
    public class CNNModel
    {
        private Conv2D conv;
        private Dense fc;

        public CNNModel()
        {
            conv = new Conv2D(1, 4, 3);    // 단순 구조
            fc = new Dense(4 * 26 * 26, 10);
        }

        public (float[] logits, float[,,] convOut) Forward(float[,,] input)
        {
            var x = conv.Forward(input);
            var flat = Flatten.Forward(x);
            var outp = fc.Forward(flat);
            return (Activation.Softmax(outp), x);
        }

        public void Train(float[,,] input, int label, float lr)
        {
            var (pred, convOut) = Forward(input);
            float[] dLoss = new float[pred.Length];

            for (int i = 0; i < pred.Length; i++)
                dLoss[i] = pred[i] - (i == label ? 1f : 0f); // dCrossEntropy(Softmax)

            var dFc = fc.Backward(dLoss, lr);
            var dConvIn = To3D(dFc, 4, 26, 26);
            conv.Backward(dConvIn, lr);
        }

        private float[,,] To3D(float[] flat, int c, int h, int w)
        {
            float[,,] x = new float[c, h, w];
            int idx = 0;
            for (int i = 0; i < c; i++)
                for (int y = 0; y < h; y++)
                    for (int z = 0; z < w; z++)
                        x[i, y, z] = flat[idx++];
            return x;
        }
    }
}
