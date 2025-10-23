using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST.Tensor
{
    [Serializable]
    public class Tensor1D
    {
        public float[] Data;
        public int Range0;

        public Tensor1D(int range0)
        {
            Data = new float[range0];
            Range0 = range0;
        }

        public void Set(int r0, float value)
        {
            Data[r0] = value;
        }

        public float Get(int r0)
        {
            return Data[r0];
        }

        public int IndexOfMax() 
        {
            int maxIndex = 0;
            float maxValue = Data[0];
            for (int i = 1; i < Range0; i++)
            {
                if (Data[i] > maxValue)
                {
                    maxValue = Data[i];
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        public void ArrayClear()
        {
            Array.Clear(Data, 0, Data.Length);
        }
    }

    [Serializable]
    public class Tensor2D
    {
        public float[] Data;
        public int Range0;
        public int Range1;

        public Tensor2D(int range0, int range1) 
        { 
            Data = new float[range0 * range1];
            Range0 = range0;
            Range1 = range1;
        }

        public void Set(int r0, int r1, float value)
        {
            Data[r0 * Range1 + r1] = value;
        }

        public float Get(int r0, int r1)
        {
            return Data[r0 * Range1 + r1];
        }

        public void ArrayClear()
        {
            Array.Clear(Data, 0, Data.Length);
        }
    }

    [Serializable]
    public class Tensor3D
    {
        public float[] Data;
        public int Range0;
        public int Range1;
        public int Range2;

        public Tensor3D(int range0, int range1, int range2)
        {
            Data = new float[range0 * range1 * range2];
            Range0 = range0;
            Range1 = range1;
            Range2 = range2;
        }

        public void Set(int r0, int r1, int r2, float value)
        {
            Data[r0 * Range1 * Range2 + r1 * Range2 + r2] = value;
        }

        public float Get(int r0, int r1, int r2)
        {
            return Data[r0 * Range1 * Range2 + r1 * Range2 + r2];
        }

        public void ArrayClear()
        {
            Array.Clear(Data, 0, Data.Length);
        }
    }

    [Serializable]
    public class Tensor4D
    {

        public float[] Data;
        public int Range0;
        public int Range1;
        public int Range2;
        public int Range3;

        public Tensor4D(int range0, int range1, int range2, int range3)
        {
            Data = new float[range0 * range1 * range2 * range3];
            Range0 = range0;
            Range1 = range1;
            Range2 = range2;
            Range3 = range3;
        }

        public void Set(int r0, int r1, int r2, int r3, float value)
        {
            Data[r0 * Range1 * Range2 * Range3 + r1 * Range2 * Range3 + r2 * Range3 + r3] = value;
        }

        public float Get(int r0, int r1, int r2, int r3)
        {
            return Data[r0 * Range1 * Range2 * Range3 + r1 * Range2 * Range3 + r2 * Range3 + r3];
        }

        public void ArrayClear()
        {
            Array.Clear(Data, 0, Data.Length);
        }
    }
}
