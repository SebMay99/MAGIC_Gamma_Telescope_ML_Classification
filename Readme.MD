# MAGIC Gamma Telescope — ML Classification

Binary classification on the [MAGIC Gamma Telescope dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) from UCI. The goal is to distinguish gamma ray signals from hadronic background noise using several supervised learning approaches, ending with a tuned neural network.

## The Problem

Ground-based Cherenkov telescopes detect high-energy gamma rays by capturing the light patterns they produce in the atmosphere. The challenge is that most detected events are actually hadrons (background noise), not gamma rays. This dataset contains 19,020 simulated events described by 10 shape and orientation features derived from those light patterns.

The two classes are:

- `g` — gamma (signal)
- `h` — hadron (background)

## Dataset

**Source:** `magic04.data` from UCI Machine Learning Repository

**Features** (all continuous):

| Feature | Description |
|---|---|
| fLength | Major axis of ellipse |
| fWidth | Minor axis of ellipse |
| fSize | Sum of content of all pixels |
| fConc | Ratio of sum of two highest pixels over fSize |
| fConc1 | Ratio of highest pixel over fSize |
| fAsym | Distance from highest pixel to center, projected onto major axis |
| fM3Long | 3rd root of third moment along major axis |
| fM3Trans | 3rd root of third moment along minor axis |
| fAlpha | Angle of major axis with vector to origin |
| fDist | Distance from origin to center of ellipse |

## Approach

The dataset is split 60/20/20 into train, validation, and test sets. Features are standardized with `StandardScaler`. The training set is oversampled with `RandomOverSampler` to handle class imbalance before fitting.

Five classifiers are trained and evaluated:

1. **k-Nearest Neighbors** — baseline, k=5
2. **Naive Bayes** — Gaussian, assumes feature independence
3. **Logistic Regression** — linear decision boundary
4. **SVM** — RBF kernel via `SVC`
5. **Neural Network** — 2-hidden-layer MLP with hyperparameter grid search

For the neural network, a grid search is run over:
- Nodes per layer: `[16, 32, 64]`
- Dropout: `[0, 0.2]`
- Learning rate: `[0.1, 0.005, 0.001]`
- Batch size: `[32, 64, 128]`

The model with the lowest validation loss is selected and evaluated on the held-out test set.

## Stack

- Python 3
- pandas, numpy
- scikit-learn, imbalanced-learn
- TensorFlow / Keras
- matplotlib

## Running It

Open in Colab or run locally:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow
```

Download `magic04.data` from the [UCI repository](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) and place it in the same directory as the notebook.

## Results

All models are evaluated on the same held-out test set (3,804 examples). Class `0` = hadron, class `1` = gamma.

### k-Nearest Neighbors (k=5)

```
              precision    recall  f1-score   support

           0       0.73      0.72      0.73      1336
           1       0.85      0.86      0.85      2468

    accuracy                           0.81      3804
   macro avg       0.79      0.79      0.79      3804
weighted avg       0.81      0.81      0.81      3804
```

### Naive Bayes (Gaussian)

```
              precision    recall  f1-score   support

           0       0.70      0.42      0.52      1336
           1       0.74      0.90      0.81      2468

    accuracy                           0.73      3804
   macro avg       0.72      0.66      0.67      3804
weighted avg       0.73      0.73      0.71      3804
```

### Logistic Regression

```
              precision    recall  f1-score   support

           0       0.68      0.71      0.70      1336
           1       0.84      0.82      0.83      2468

    accuracy                           0.78      3804
   macro avg       0.76      0.77      0.76      3804
weighted avg       0.79      0.78      0.78      3804
```

### SVM (RBF Kernel)

```
              precision    recall  f1-score   support

           0       0.81      0.79      0.80      1336
           1       0.89      0.90      0.90      2468

    accuracy                           0.86      3804
   macro avg       0.85      0.85      0.85      3804
weighted avg       0.86      0.86      0.86      3804
```

### Neural Network (best validation loss)

```
              precision    recall  f1-score   support

           0       0.81      0.82      0.82      1336
           1       0.90      0.90      0.90      2468

    accuracy                           0.87      3804
   macro avg       0.86      0.86      0.86      3804
weighted avg       0.87      0.87      0.87      3804
```

### Summary

| Model | Accuracy | Gamma F1 | Hadron F1 |
|---|---|---|---|
| Naive Bayes | 0.73 | 0.81 | 0.52 |
| Logistic Regression | 0.78 | 0.83 | 0.70 |
| kNN (k=5) | 0.81 | 0.85 | 0.73 |
| SVM (RBF) | 0.86 | 0.90 | 0.80 |
| Neural Network | 0.87 | 0.90 | 0.82 |

SVM and the neural network perform comparably. The main gap across all models is on the hadron class — correctly rejecting background events is harder than identifying signal, which is typical for this dataset.
