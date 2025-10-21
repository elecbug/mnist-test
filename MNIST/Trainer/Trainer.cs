using MNIST.Network;

namespace MNIST.Trainer
{
    public static class Trainer
    {
        public static void Train(CNNModel model, List<float[,,]> trainImages, List<int> trainLabels, int epochs, float lr)
        {
            for (int e = 0; e < epochs; e++)
            {
                float loss = 0;
                int correct = 0;
                for (int i = 0; i < trainImages.Count; i++)
                {
                    model.Train(trainImages[i], trainLabels[i], lr);
                    var (pred, _) = model.Forward(trainImages[i]);
                    int predicted = Array.IndexOf(pred, pred.Max());
                    if (predicted == trainLabels[i]) correct++;

                    // CrossEntropy loss
                    loss += -MathF.Log(pred[trainLabels[i]] + 1e-8f);
                }

                Console.WriteLine($"Epoch {e + 1} | Loss: {loss / trainImages.Count:F4} | Accuracy: {(float)correct / trainImages.Count * 100:F2}%");
            }
        }
    }
}
