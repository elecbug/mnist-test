using MNIST.Tensor;

namespace MNIST.Layer
{
    [Serializable]
    public class Dropout
    {
        private float rate;
        private bool[] mask = { };

        public Dropout(float rate = 0.5f)
        {
            this.rate = rate;
        }

        public Tensor1D Forward(Tensor1D input, bool training = true)
        {
            mask = new bool[input.Range0];
            Tensor1D output = new Tensor1D(input.Range0);
            for (int i = 0; i < input.Range0; i++)
            {
                mask[i] = training ? new Random().NextDouble() > rate : true;
                output.Set(i, mask[i] ? input.Get(i) / (1f - rate) : 0f);
            }
            return output;
        }

        public Tensor1D Backward(Tensor1D dOut)
        {
            Tensor1D dInput = new Tensor1D(dOut.Range0);
            for (int i = 0; i < dOut.Range0; i++)
                dInput.Set(i, mask[i] ? dOut.Get(i) / (1f - rate) : 0f);
            return dInput;
        }
    }
}
