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
        private PictureBox InvertedPBox { get; set; }
        private Label PredictLabel { get; set; }
        private Label[] PredictLabels { get; set; }
        private ProgressBar[] PredictBars { get; set; }

        private bool formClosing = false;

        public MainForm()
        {
            {
                Text = "MNIST Tester";
                Width = 800;
                Height = 800;
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

            InvertedPBox = new PictureBox()
            {
                Parent = this,
                Top = 100,
                Left = 400,
                Width = 280,
                Height = 280,
                BorderStyle = BorderStyle.FixedSingle,
                SizeMode = PictureBoxSizeMode.StretchImage,
            };

            PredictLabel = new Label()
            {
                Parent = this,
                Text = "Draw a digit and use Convert to predict.",
                AutoSize = true,
                Top = 400,
                Left = 50,
            };
            PredictLabels = new Label[10];
            PredictBars = new ProgressBar[10];

            for (int i = 0; i < 10; i++)
            {
                PredictLabels[i] = new Label()
                {
                    Parent = this,
                    Text = $"{i}: ",
                    AutoSize = true,
                    Top = 430 + i * 25,
                    Left = 50,
                };

                PredictBars[i] = new ProgressBar()
                {
                    Parent = this,
                    Top = 430 + i * 25,
                    Left = 80,
                    Width = 600,
                    Height = 20,
                    Minimum = 0,
                    Maximum = 100,
                };
            }

            MainMenu.Items.Add("Load Model", null, (s, e) => LoadModel());
            MainMenu.Items.Add("Train Model", null, (s, e) => TrainModel());
            MainMenu.Items.Add("Test Model", null, (s, e) => TestModel());
            MainMenu.Items.Add("Clear Canvas", null, (s, e) => DrawingCanvas.ClearCanvas());

            new Thread(Convert).Start();

            FormClosing += (s, e) =>
            {
                formClosing = true;
            };
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

            int start = new Random().Next(0, 60000);
            int end = Math.Min(start + 10000, 60000)- start;

            MnistTrainer.Train(CnnModel, trainImages.GetRange(start, end),
                trainLabels.GetRange(start, end), epochs: 5, batchSize: 32, optimizer: optimizer);

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
            while (!formClosing)
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
                        PredictLabel.Text = $"Predicted Digit: {predicted}\r\n";

                        for (int i = 0; i < 10; i++)
                        {
                            PredictBars[i].Value = (int)(pred[i] * 100f);
                        }

                        InvertedPBox.Image = DrawingCanvas.Inverted;
                    });

                    drawed = false;
                }
            }
        }
    }
}
