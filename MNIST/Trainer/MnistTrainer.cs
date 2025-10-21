using MNIST.Network;
using System.Diagnostics;
using static MNIST.Network.Optimizer;

namespace MNIST.Trainer
{
    [Serializable]
    public static class MnistTrainer
    {
        public static void Train(
            CNNModel model,
            List<float[,,]> trainImages,
            List<int> trainLabels,
            int epochs,
            int batchSize,
            Adam optimizer)
        {
            int total = trainImages.Count;
            var rand = new Random();

            for (int e = 0; e < epochs; e++)
            {
                Debug.WriteLine($"Starting epoch {e + 1}/{epochs}...");
                float totalLoss = 0;
                int correct = 0;

                var indices = Enumerable.Range(0, total).OrderBy(_ => rand.Next()).ToList();

                for (int start = 0; start < total; start += batchSize)
                {
                    int end = Math.Min(start + batchSize, total);
                    int batchCount = end - start;

                    model.ZeroGrad();
                    float batchLoss = 0;

                    for (int b = start; b < end; b++)
                    {
                        int idx = indices[b];
                        var img = trainImages[idx];
                        var label = trainLabels[idx];

                        var (pred, _) = model.Forward(img, training: true);

                        int predLabel = Array.IndexOf(pred, pred.Max());
                        if (predLabel == label) correct++;
                        batchLoss += -MathF.Log(pred[label] + 1e-8f);

                        float[] dLoss = new float[pred.Length];
                        for (int i = 0; i < pred.Length; i++)
                            dLoss[i] = pred[i] - (i == label ? 1f : 0f); 

                        model.Backward(dLoss, lr: optimizer.LearningRate);
                    }

                    model.Step(optimizer, 1f);

                    totalLoss += batchLoss / batchCount; 
                }

                float avgLoss = totalLoss / (total / batchSize);
                float acc = (float)correct / total * 100f;
                Debug.WriteLine($"Epoch {e + 1} | Loss: {avgLoss:F4} | Acc: {acc:F2}%");
            }
        }
    }
}
