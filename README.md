# Fashion MNIST Image Classification with a Six-Layer CNN (Keras, Python and R)

**Author:** Moses
**Context:** Junior Machine Learning Researcher assignment, Microsoft AI. The goal is to build and understand a CNN on Fashion MNIST, then adapt the same code later to classify user profile images for targeted marketing.

---

## 1. What this project does

1. Loads the Fashion MNIST dataset (70,000 grayscale 28x28 images of clothing, 10 classes).
2. Explores and visualizes the data.
3. Preprocesses it (normalization, channel axis, train/validation split).
4. Builds a **six-layer CNN** inside a Python class (`FashionCNN`) using Keras.
5. Trains and evaluates it, with learning curves and a confusion matrix.
6. Predicts individual images (at least two) and shows the model's confidence.
7. Repeats the CNN in **R** using the `keras` package.

## 2. Project structure

```
fashion-mnist-cnn/
├── README.md
├── requirements.txt
├── python/
│   └── fashion_mnist_cnn.ipynb      # Python notebook, run top to bottom
├── r/
│   └── fashion_mnist_cnn.R          # R script (or .Rmd)
└── images/
    ├── sample_grid.png              # 25 example images with labels
    ├── training_curves.png          # accuracy and loss per epoch
    ├── confusion_matrix.png         # test-set confusion matrix
    └── prediction_0.png, prediction_1.png, ...   # individual predictions
```

> Adjust file names above to match what you actually submit.

## 3. Setup

### Python

Tested with Python 3.x and TensorFlow `<fill in: print(tf.__version__)>`.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:

```
tensorflow
numpy
matplotlib
scikit-learn
jupyter
```

Run the notebook:

```bash
jupyter notebook python/fashion_mnist_cnn.ipynb
```

Then choose **Kernel > Restart & Run All**. Running cells out of order or twice can cause shape errors, because notebooks keep old variable values.

### R

Tested with R `<fill in: R.version.string>`.

```r
install.packages(c("keras3", "tensorflow"))   # or "keras" for older setups
library(keras3)
install_keras()                               # installs the Python backend
```

Then run `r/fashion_mnist_cnn.R`.

> Fill in the exact package names and commands you used, since the R packages have changed over time.

## 4. Data

- **Dataset:** Fashion MNIST, created by Zalando Research.
- **Loaded with:** `keras.datasets.fashion_mnist.load_data()` (downloaded automatically on first run).
- **Size:** 60,000 training images and 10,000 test images. Each image is 28x28 pixels, grayscale, with integer pixel values from 0 to 255.
- **Classes (6,000 training images each, so perfectly balanced):**

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

The class names are not stored in the dataset. They follow the official Fashion MNIST documentation by Zalando Research, and were checked visually against the sample images.

## 5. Preprocessing

| Step | What | Why |
|------|------|-----|
| Normalize | `astype('float32') / 255.0` | Scales pixels from 0-255 to 0-1 for faster, more stable training |
| Add channel axis | shape `(N, 28, 28)` to `(N, 28, 28, 1)` | `Conv2D` expects a channel dimension (1 = grayscale) |
| Validation split | first 5,000 training images held out | Used to monitor overfitting; the test set stays untouched until the end |

Final shapes: train `(55000, 28, 28, 1)`, validation `(5000, 28, 28, 1)`, test `(10000, 28, 28, 1)`. The test set gets exactly the same preprocessing as the training set.

## 6. Model architecture

**Layer-counting convention:** six layers = 2 `Conv2D` + 2 `MaxPooling2D` + 2 `Dense`. `Flatten` only reshapes data and is not counted.

| # | Layer | Output shape | Parameters |
|---|-------|--------------|-----------|
| 1 | Conv2D, 32 filters, 3x3, ReLU | (26, 26, 32) | 320 |
| 2 | MaxPooling2D, 2x2 | (13, 13, 32) | 0 |
| 3 | Conv2D, 64 filters, 3x3, ReLU | (11, 11, 64) | 18,496 |
| 4 | MaxPooling2D, 2x2 | (5, 5, 64) | 0 |
| - | Flatten | (1600) | 0 |
| 5 | Dense, 128 units, ReLU | (128) | 204,928 |
| 6 | Dense, 10 units, Softmax | (10) | 1,290 |

**Total trainable parameters: 225,034.**

The model is wrapped in a class, `FashionCNN`, which takes `input_shape` and `num_classes` as arguments. To adapt it to profile images, change those two values (for example `input_shape=(64, 64, 3)` for 64x64 color images) instead of rewriting the architecture.

Class methods: `compile()`, `train()`, `plot_history()`, `evaluate()`, `predict_image()`, `summary()`.

## 7. Training setup

| Setting | Value | Reason |
|---------|-------|--------|
| Loss | `sparse_categorical_crossentropy` | Labels are integers 0-9, not one-hot vectors |
| Optimizer | Adam, learning rate 0.001 | Adapts step sizes per weight, a good default |
| Metric | Accuracy | Classes are balanced, so accuracy is a fair measure |
| Epochs | 10 | `<edit if you changed it>` |
| Batch size | 64 | `<edit if you changed it>` |

**Important order of operations:** build the object, then `compile()`, then `train()`. A freshly created `FashionCNN()` must be compiled again before training or evaluating.

```python
cnn = FashionCNN()
cnn.compile()
history = cnn.train(x_train, y_train, x_val, y_val, epochs=10)
cnn.plot_history()
cnn.evaluate(x_test, y_test)
```

## 8. Results

> Fill these in from your own run. Results vary slightly between runs because weights start randomly.

| Metric | Value |
|--------|-------|
| Final training accuracy | `<fill in>` |
| Final validation accuracy | `<fill in>` |
| **Test accuracy** | `<fill in>` |
| Test loss | `<fill in>` |

**Learning curves:** see `images/training_curves.png`. `<Describe in one or two sentences whether training and validation tracked each other, or whether overfitting appeared and at which epoch.>`

**Confusion matrix:** see `images/confusion_matrix.png`. `<State which classes were confused most, for example Shirt vs T-shirt/top, and which class had the lowest F1 score.>`

## 9. Predictions

At least two test images are classified, each shown with the predicted label, the model's confidence, the true label, and a bar chart of all 10 class probabilities. A green title means correct and red means wrong.

| Test image index | Predicted | Confidence | Actual | Correct? |
|------------------|-----------|------------|--------|----------|
| `<0>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| `<1>` | `<fill in>` | `<fill in>` | `<fill in>` | `<fill in>` |
| `<a mistake>` | `<fill in>` | `<fill in>` | `<fill in>` | No |

Images are saved as `images/prediction_*.png`.

To predict a single image yourself:

```python
cnn.predict_image(x_test[0], true_label=y_test[0], class_names=class_names)
```

A single image must be given a batch axis, shape `(1, 28, 28, 1)`, before it goes into the model. `predict_image` does this for you.

## 10. R implementation

`<Complete this section once the R version is built.>` It should state:

- The R and package versions used.
- That the same architecture (six layers, same filters and units), preprocessing, loss, optimizer, and data split were used.
- The test accuracy from R, compared with the Python result.
- How to run it: `Rscript r/fashion_mnist_cnn.R`, or open the file in RStudio and run it top to bottom.

## 11. Reproducing the results

1. Install the dependencies (section 3).
2. Open `python/fashion_mnist_cnn.ipynb` and choose **Restart & Run All**.
3. Figures are written to `images/`.
4. Run the R script for the R version.

## 12. Notes and limitations

- Pixel values and shapes are checked at each step (`.shape`, `.min()`, `.max()`), so any preprocessing mismatch shows up early.
- The test set is used only once, for the final score, and never to make design decisions.
- Fashion MNIST images are small grayscale images on a clean background. Profile photos are larger, in color, and much more varied, so expect to change the input shape, add data augmentation, and probably use a deeper or pretrained model when adapting this work.

## 13. References

- Fashion MNIST: Zalando Research, https://github.com/zalandoresearch/fashion-mnist
- Keras documentation: https://keras.io
