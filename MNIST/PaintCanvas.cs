using MNIST.Tensor;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MNIST
{
    public class PaintCanvas : Panel
    {
        public bool Drawing { get; private set; } = false;
        public Bitmap? Inverted { get; private set; }
        private Bitmap canvas;
        private Point lastPoint;
        private Color nowColor = Color.Black;
        private Color clearColor = Color.White;

        public Tensor3D ToMnistInput()
        {
            int size = 28;
            using Bitmap bmp = new Bitmap(size, size);

            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.DrawImage(canvas, 0, 0, size, size);
            }

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    Color c = bmp.GetPixel(x, y);
                    Color inv = Color.FromArgb(255 - c.R, 255 - c.G, 255 - c.B);
                    bmp.SetPixel(x, y, inv);
                }
            }

            bmp.Save("debug_inverted.png"); 

            int minX = size, minY = size, maxX = 0, maxY = 0;
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    Color c = bmp.GetPixel(x, y);
                    float gray = (c.R + c.G + c.B) / 3f / 255f;
                    if (gray > 0.05f)
                    {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            if (minX >= maxX || minY >= maxY)
            {
                return new Tensor3D(1, size, size);
            }

            int w = maxX - minX + 1;
            int h = maxY - minY + 1;

            using Bitmap cropped = new Bitmap(w, h);
            using (Graphics g2 = Graphics.FromImage(cropped))
            {
                g2.DrawImage(bmp, 0, 0, new Rectangle(minX, minY, w, h), GraphicsUnit.Pixel);
            }

            using Bitmap centered = new Bitmap(size, size);
            using (Graphics g3 = Graphics.FromImage(centered))
            {
                g3.Clear(Color.Black);
                float scale = 20f / Math.Max(w, h);
                int newW = (int)(w * scale);
                int newH = (int)(h * scale);

                int offsetX = (size - newW) / 2;
                int offsetY = (size - newH) / 2 - 1;

                g3.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g3.SmoothingMode = SmoothingMode.HighQuality;
                g3.DrawImage(cropped, offsetX, offsetY, newW, newH);
            }

            centered.Save("debug_centered_mnist_style.png");
            Inverted = new Bitmap(centered);

            float gamma = 1f; 
            Tensor3D mnistInput = new Tensor3D(1, size, size);

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    Color c = centered.GetPixel(x, y);
                    float gray = (c.R + c.G + c.B) / 3f / 255f;

                    float boosted = MathF.Pow(gray, gamma);
                    mnistInput.Set(0, y, x, boosted);
                }
            }

            return mnistInput;
        }

        public void ClearCanvas()
        {
            using Graphics g = Graphics.FromImage(canvas);
            g.Clear(clearColor);
            this.Invalidate();
        }

        public PaintCanvas(int width, int height)
        {
            this.Width = width;
            this.Height = height;

            canvas = new Bitmap(width, height);
            ClearCanvas();

            this.MouseDown += (s, e) => { Drawing = true; lastPoint = e.Location; nowColor = e.Button == MouseButtons.Left ? Color.Black : Color.White; };
            this.MouseUp += (s, e) => { Drawing = false; };
            this.MouseMove += (s, e) =>
            {
                if (!Drawing) return;

                using Graphics g = Graphics.FromImage(canvas);
                int penSize = 10;
                Brush brush = new SolidBrush(nowColor);
                g.DrawEllipse(new Pen(brush, penSize), 
                    e.Location.X - penSize / 2, e.Location.Y - penSize / 2, penSize, penSize);
                g.DrawLine(new Pen(brush, penSize * 2), lastPoint, e.Location);
                lastPoint = e.Location;
                this.Invalidate();
            };

            this.Paint += (s, e) =>
            {
                lock (this)
                    e.Graphics.DrawImage(canvas, 0, 0);
            };
        }
    }
}
