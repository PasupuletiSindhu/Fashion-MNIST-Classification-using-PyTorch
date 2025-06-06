# Fashion MNIST Classification using PyTorch

This project explores the use of deep learning models for image classification on the Fashion MNIST dataset using PyTorch. The goal is to compare model performance with standard Cross Entropy Loss versus Focal Loss and also explore the use of CNN architectures for improved accuracy.

---

## Project Structure

```
├── Main.ipynb      		     # Main notebook containing all experiments
├── model.pth                        # Saved model trained with Cross Entropy Loss
├── model_focal_loss.pth             # Saved model trained with Focal Loss
├── cnn_model.pth                    # Saved CNN model (likely trained with Focal Loss)
```

---

## Dataset

We use the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), a drop-in replacement for the original MNIST dataset. It contains 60,000 training and 10,000 test grayscale images of 10 different categories of fashion items.

---

## Models & Techniques

### 1. **Baseline Feedforward Neural Network**
- Architecture:
  - Input → Linear(784→512) → ReLU
  - Linear(512→512) → ReLU
  - Linear(512→10)
- Optimizer: SGD
- Loss: Cross Entropy

### 2. **Focal Loss Integration**
- Designed to down-weight easy examples and focus training on hard negatives.
- Parameters:
  - Gamma = 2.0
  - Alpha = 1.0

### 3. **CNN Model (Work In Progress or Completed)**
- Uses a convolutional architecture with a 3x3 kernel.
- Model file: `cnn_model.pth`

---

## Training and Evaluation

| Model            | Loss Function     | Final Accuracy | Validation Loss |
|------------------|-------------------|----------------|------------------|
| Feedforward NN   | Cross Entropy     | 64.6%          | 1.1040           |
| Feedforward NN   | Focal Loss        | 67.8%          | 0.5404           |

Visualizations include training and validation loss curves, and a single test sample prediction with class label comparison.

---

## Observations

- Focal Loss significantly improved the learning dynamics.
- The model trained with Focal Loss converged faster and generalized better on validation data.
- CNN implementation (code not shown in first few cells) is expected to further improve results by leveraging spatial features.

---

## Requirements

```bash
pip install torch torchvision matplotlib
```

---

## Run Instructions

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/Sindhu_Pasupuleti_HW7.git
   cd Sindhu_Pasupuleti_HW7
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook HW7_Sindhu_Pasupuleti.ipynb
   ```

3. To use saved models:
   ```python
   model.load_state_dict(torch.load("model.pth"))
   ```

---

## Author

**Sindhu Pasupuleti**

---

## 📌 Acknowledgements

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- Fashion MNIST dataset by Zalando Research

---
