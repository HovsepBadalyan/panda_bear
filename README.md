# 🐼 Panda vs Bear Image Classification — PyTorch CNN

A binary image classification project built from scratch using **PyTorch**. A custom convolutional neural network (CNN) predicts whether an image contains a **bear** or a **panda**.

## Results

| Metric | Result |
|---|---:|
| Test accuracy | **100.00% (100/100)** |
| Test loss | **0.0087** |
| Final training accuracy | 99.00% |
| Final validation accuracy | 99.00% |
| Training epochs | 15 |
| Batch size | 32 |
| Optimizer | Adam (learning rate: 0.001) |
| Loss function | CrossEntropyLoss |

> These results are from the provided 100-image test split. Perfect accuracy on a small test set does **not** guarantee equivalent performance on arbitrary real-world images. Additional independent testing and a check for duplicate or closely related images across splits are recommended.

## Dataset

Source: [Panda or Bear Image Classification — Kaggle](https://www.kaggle.com/datasets/mattop/panda-or-bear-image-classification)

- **500** images in the original training folder, split into **400 training** and **100 validation** images.
- **100** images in the held-out test folder.
- Classes are assigned alphabetically by `torchvision.datasets.ImageFolder`:
  - `0` — Bears
  - `1` — Pandas

Expected directory layout:

```text
PandasBears/
├── Train/
│   ├── Bears/
│   └── Pandas/
├── Test/
│   ├── Bears/
│   └── Pandas/
└── Untitled(3).ipynb
```

The notebook uses relative paths `Train` and `Test`, so run it with the working directory set to `PandasBears`.

## Model architecture

A custom CNN implemented with `torch.nn.Module`, without pretrained weights:

```text
Input image                 [3, 224, 224]
Conv2d(3, 32, 3) + ReLU
MaxPool2d(2)                [32, 112, 112]
Conv2d(32, 64, 3) + ReLU
MaxPool2d(2)                [64, 56, 56]
Conv2d(64, 128, 3) + ReLU
MaxPool2d(2)                [128, 28, 28]
AdaptiveAvgPool2d(1, 1)     [128, 1, 1]
Flatten                     [128]
Linear(128, 64) + ReLU
Dropout(0.3)
Linear(64, 2)               [2 logits]
```

The final layer outputs **logits**, not probabilities. `CrossEntropyLoss` consumes these logits directly; `softmax` is used only when reporting prediction confidence.

## Preprocessing

- Resize images to **224 × 224**.
- Apply random horizontal flip (`p=0.5`) **only to training images**.
- Convert RGB images to tensors.
- Normalize RGB channels with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.
- Create reproducible training/validation index splits with random seed `42`.

## Setup and usage

1. Install Python and Jupyter Notebook.
2. Install dependencies:

   ```bash
   pip install torch torchvision matplotlib scikit-learn pillow notebook
   ```

3. Download and extract the dataset from Kaggle into the directory layout shown above.
4. Place the notebook in the `PandasBears` directory and launch Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `Untitled(3).ipynb` and run its cells in order to prepare data, train the CNN, evaluate it, and make predictions.

The notebook automatically uses CUDA when available, otherwise CPU.

## Predicting a new image

After running the training cells in the notebook:

```python
predict_image("panda.jpg")
predict_image("bear.jpg")
```

The `predict_image(image_path)` function displays the input image, predicted class, and softmax confidence, and returns `(class_name, confidence_percentage)`.

Example outputs observed in the notebook:

```text
Prediction: Pandas
Confidence: 99.99%

Prediction: Bears
Confidence: 100.00%
```

Confidence is the model's softmax score and should not be interpreted as a calibrated probability of correctness.

## Technologies

- Python
- PyTorch (`torch`, `torchvision`)
- Pillow
- Matplotlib
- scikit-learn (installed for additional evaluation)
- Jupyter Notebook

## Potential improvements

- Evaluate on a larger, independently sourced dataset.
- Check for duplicates or near-duplicates across training, validation, and test splits.
- Save and load model weights with `torch.save` and `torch.load`.
- Add a confusion matrix, classification report, and learning-curve plots.
- Compare the custom CNN with a pretrained ResNet18.
- Build a small Flask or Streamlit interface for uploading images.

## Notes

This is an educational computer-vision project. The notebook currently trains the model in memory; it does not yet include a model checkpoint export or a standalone inference application.

## Dataset attribution

Dataset by **mattop**, hosted on Kaggle: [Panda or Bear Image Classification](https://www.kaggle.com/datasets/mattop/panda-or-bear-image-classification). Check the dataset page for its applicable usage terms before redistributing images.
