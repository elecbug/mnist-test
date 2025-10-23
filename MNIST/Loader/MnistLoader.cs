using MNIST.Tensor;

namespace MNIST.Loader
{
    [Serializable]
    public static class MnistLoader
    {
        private static int ReverseBytes(int v)
        {
            byte[] bytes = BitConverter.GetBytes(v);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static (List<Tensor3D> images, List<int> labels) Load(string imgPath, string lblPath)
        {
            using var imgFs = new FileStream(imgPath, FileMode.Open);
            using var lblFs = new FileStream(lblPath, FileMode.Open);
            using var imgBr = new BinaryReader(imgFs);
            using var lblBr = new BinaryReader(lblFs);

            imgBr.ReadInt32(); // magic
            int numImages = ReverseBytes(imgBr.ReadInt32());
            int numRows = ReverseBytes(imgBr.ReadInt32());
            int numCols = ReverseBytes(imgBr.ReadInt32());
            lblBr.ReadInt32();
            lblBr.ReadInt32();

            var images = new List<Tensor3D>(numImages);
            var labels = new List<int>(numImages);

            for (int i = 0; i < numImages; i++)
            {
                Tensor3D img = new Tensor3D(1, numRows, numCols); // [channel, height, width]
                for (int r = 0; r < numRows; r++)
                    for (int c = 0; c < numCols; c++)
                        img.Set(0, r, c, imgBr.ReadByte() / 255f);

                images.Add(img);
                labels.Add(lblBr.ReadByte());
            }
            return (images, labels);
        }
    }
}
