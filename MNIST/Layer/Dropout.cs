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

        public float[] Forward(float[] input, bool training = true)
        {


            mask = new bool[input.Length];
            float[] output = new float[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                mask[i] = training ? new Random().NextDouble() > rate : true;
                output[i] = mask[i] ? input[i] / (1f - rate) : 0f;
            }
            return output;
        }

        public float[] Backward(float[] dOut)
        {
            float[] dInput = new float[dOut.Length];
            for (int i = 0; i < dOut.Length; i++)
                dInput[i] = mask[i] ? dOut[i] / (1f - rate) : 0f;
            return dInput;
        }
    }
}
