using MNIST.Loader;
using MNIST.Network;
using MNIST.Trainer;
using System.Diagnostics;
using System.Reflection;
using static MNIST.Network.Optimizer;

namespace MNIST
{
    public class MainForm : Form
    {
        private MenuStrip MainMenu { get; set; }
        private CNNModel? CnnModel { get; set; }
        private Label SelectedModelLabel { get; set; }
        private PaintCanvas DrawingCanvas { get; set; }
        private Label PredicteLabel { get; set; }

        public MainForm()
        {
            {
                Width = 800;
                Height = 600;
            }

            MainMenu = new MenuStrip()
            {
                Parent = this,
                Visible = true,
            };

            SelectedModelLabel = new Label()
            {
                Parent = this,
                Text = "No model loaded.",
                AutoSize = true,
                Top = 50,
                Left = 50,
            };

            DrawingCanvas = new PaintCanvas(280, 280)
            {
                Parent = this,
                Top = 100,
                Left = 50,
                BorderStyle = BorderStyle.FixedSingle,
            };

            PredicteLabel = new Label()
            {
                Parent = this,
                Text = "Draw a digit and use Convert to predict.",
                AutoSize = true,
                Top = 400,
                Left = 50,
            };

            MainMenu.Items.Add("Load Model", null, (s, e) => LoadModel());
            MainMenu.Items.Add("Train Model", null, (s, e) =>TrainModel());
            MainMenu.Items.Add("Test Model", null, (s, e) =>TestModel());
            MainMenu.Items.Add("Clear Canvas", null, (s, e) => DrawingCanvas.ClearCanvas());

            new Thread(Convert).Start();
        }

        private void LoadModel()
        {
            OpenFileDialog ofd = new OpenFileDialog()
            {
                Filter = "Binary Files|*.bin",
                Title = "Load CNN Model",
            };

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                CnnModel = CNNModel.Load(ofd.FileName);
                SelectedModelLabel.Text = $"Loaded Model: {Path.GetFileName(ofd.FileName)}";

                MessageBox.Show("Model loaded successfully!", "Info",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private void TrainModel()
        {
            if (CnnModel == null)
                CnnModel = new CNNModel();

            var optimizer = new Adam(learningRate: 0.0003f);

            var (trainImages, trainLabels) = MnistLoader.Load(
            "dataset/train-images.idx3-ubyte",
            "dataset/train-labels.idx1-ubyte");

            var (testImages, testLabels) = MnistLoader.Load(
                "dataset/t10k-images.idx3-ubyte",
                "dataset/t10k-labels.idx1-ubyte");

            MnistTrainer.Train(CnnModel, trainImages.GetRange(0, 2000),
                trainLabels.GetRange(0, 2000), epochs: 5, batchSize: 32, optimizer: optimizer);

            SaveFileDialog sfd = new SaveFileDialog()
            {
                Filter = "Binary Files|*.bin",
                Title = "Save CNN Model After Training",
            };

            if (sfd.ShowDialog() == DialogResult.OK)
            {
                CnnModel.Save(sfd.FileName);
                SelectedModelLabel.Text = $"Loaded Model: {Path.GetFileName(sfd.FileName)}";

                MessageBox.Show("Training completed!", "Info",
                    MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
        }

        private void TestModel()
        {
            if (CnnModel == null)
            {
                MessageBox.Show("Please load or train a model first.", "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            var (testImages, testLabels) = MnistLoader.Load(
                "dataset/t10k-images.idx3-ubyte",
                "dataset/t10k-labels.idx1-ubyte");

            int correct = 0;
            int total = 100;

            for (int i = 0; i < total; i++)
            {
                var (pred, _) = CnnModel.Forward(testImages[i]);
                int predicted = Array.IndexOf(pred, pred.Max());

                if (predicted == testLabels[i])
                    correct++;
            }

            MessageBox.Show($"Test Accuracy: {(correct / (float)total) * 100f:F2}%", "Info",
                MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void Convert()
        {
            bool drawed = false;
            while (true)
            {
                if (CnnModel == null)
                {
                    continue;
                }
                if (DrawingCanvas.Drawing)
                {
                    drawed = true;
                    continue;
                }

                if (drawed)
                {
                    float[,,] mnistInput;

                    lock (DrawingCanvas)
                    {
                        mnistInput = DrawingCanvas.ToMnistInput();
                    }

                    var (pred, _) = CnnModel.Forward(mnistInput);
                    int predicted = Array.IndexOf(pred, pred.Max());

                    Invoke(() =>
                    {
                        PredicteLabel.Text = $"Predicted Digit: {predicted}";
                    });

                    drawed = false;
                }
            }
        }
    }
}
