using MNIST.Loader;
using MNIST.Network;
using MNIST.Trainer;
using System.Diagnostics;
using static MNIST.Network.Optimizer;

namespace MNIST
{
    public class MainForm : Form
    {
        public MainForm()
        {
            Debug.WriteLine("Loading dataset...");
            var (trainImages, trainLabels) = MnistLoader.Load(
                "dataset/train-images.idx3-ubyte",
                "dataset/train-labels.idx1-ubyte");

            CNNModel model;
            if (File.Exists("mnist_cnn_model.bin"))
                model = CNNModel.Load("mnist_cnn_model.bin");
            else
                model = new CNNModel();
            var optimizer = new Adam(learningRate: 0.0003f);
            Debug.WriteLine("Training started...");
            MnistTrainer.Train(model, trainImages.GetRange(0, 2000),
                trainLabels.GetRange(0, 2000), epochs: 5, batchSize: 32, optimizer: optimizer);
            
            var (testImages, testLabels) = MnistLoader.Load(
                "dataset/t10k-images.idx3-ubyte",
                "dataset/t10k-labels.idx1-ubyte");

            int correct = 0;
            int total = 100;
            for (int i = 0; i < total; i++)
            {
                var (pred, _) = model.Forward(testImages[i]);
                int predicted = Array.IndexOf(pred, pred.Max());
                Debug.WriteLine($"[Sample {i}] Label={testLabels[i]}, Pred={predicted}");

                if (predicted == testLabels[i])
                    correct++;
            }

            Debug.WriteLine($"\nTest Accuracy: {(float)correct / total * 100:F2}%");

            model.Save("mnist_cnn_model.bin");
        }
    }
}
