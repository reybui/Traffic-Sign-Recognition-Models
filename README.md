# Image Classification with SVM and MLP

Classical and neural baselines for multi-class image classification using:
- scikit-learn SVM (RBF kernel) with subset-based hyperparameter search
- Keras MLP (GPU-capable) with normalization, dropout, and early stopping

The data used is from: https://www.kaggle.com/datasets/flo2607/traffic-signs-classification.
The project loads images from a folder tree, preprocesses them to fixed 32×32×3 RGB, flattens to vectors, and evaluates models on a held-out test split.

## Dataset

Expected layout:
```
archive/
└── myData/
    ├── 0/
    │   ├── img1.jpg
    │   └── ...
    ├── 1/
    ├── 2/
    └── ...
```
- Each subfolder is a class label (folder name).
- Images can be grayscale/RGB/RGBA and any size; they are converted to RGB and resized.

## Features

- Auto-discovery of class folders under archive/myData
- Preprocessing: RGB conversion, resize to 32×32, scale to [0,1], flatten to 3072 features
- Caching: saves preprocessed arrays to speed up reruns
- Stratified train/test split (default test_size=0.3)
- Subset-based tuning:
  - Small subset (e.g., 20/class) for coarse GridSearch
  - Medium subset (e.g., 100/class) for refined GridSearch
- Final SVM training with best params and full evaluation
- Keras MLP (512→256) with Normalization, Dropout, EarlyStopping, ReduceLROnPlateau
- Metrics: accuracy, precision, recall, F1-score, confusion matrix
- Training vs validation accuracy plot for the MLP

## Quickstart

1) Create a virtual environment (Windows PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you don’t have a requirements.txt yet:
```
pip install numpy pandas scikit-learn scikit-image matplotlib tensorflow
```

2) Open in VS Code and run the notebook:
- File: Untitled-1.ipynb
- Run cells top to bottom.

3) Place your data under archive/myData as shown above.

## How it works

- Preprocessing:
  - Convert to 3-channel RGB, drop alpha if present
  - Resize to 32×32 with anti-aliasing
  - Normalize to [0,1] floats and flatten to vectors
- Splitting:
  - Stratified train/test to preserve class balance
- Tuning (SVM):
  - Coarse GridSearch on a small subset
  - Refined GridSearch on a medium subset
  - Final fit on full training split with chosen params
- MLP (Keras):
  - Normalization layer adapted on X_train
  - Dense(512, relu) → Dropout(0.3) → Dense(256, relu) → Dropout(0.3) → Dense(num_classes, softmax)
  - EarlyStopping and ReduceLROnPlateau
  - Evaluate on the held-out test set

## Usage notes

- Comparing models: always compare on the same held-out test split.
  - SVM: model.score(X_test, y_test)
  - Keras: model.evaluate(X_test_np, y_test_np)
- Metrics (MLP example):
  - Convert softmax probabilities to labels via argmax:
    - y_pred = model.predict(X_test_np).argmax(axis=1)
- Saving models:
  - SVM: pickle.dump(svm_model, open('model.sav','wb'))
  - Keras (recommended): model.save('MLPmodel.keras')

## Results (fill with your numbers)

| Model         | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---------------|---------------|-------------------|----------------|------------|
| SVM (RBF)     | …             | …                 | …              | …          |
| MLP (Keras)   | …             | …                 | …              | …          |

Tips:
- If the SVM wins on raw pixels, that’s common at 32×32. For better image performance, consider a small CNN.
- If training time is high, try PCA before SVM (e.g., 128–256 components) and retune C/gamma.

## GPU

- Keras uses your NVIDIA GPU automatically if TensorFlow-GPU/CUDA is installed.
- For GPU-accelerated SVM, consider RAPIDS cuML in WSL2 (optional, not required here).

## Project structure

- Untitled-1.ipynb — main notebook with preprocessing, SVM, and MLP workflows
- archive/myData — dataset root (you provide)
- data.pickle — cached preprocessed arrays (auto-created)
- model.sav — saved SVM model (auto-created)
- MLPmodel.keras — saved Keras model (recommended)

## Reproducibility

- Fixed random_state for train/test split
- Deterministic subset builders for coarse/refined searches
- Note: Some GPU ops may still introduce nondeterminism

## License

Choose a license you prefer (e.g., MIT) and add a LICENSE file.

## Acknowledgments

- scikit-learn, scikit-image, TensorFlow/Keras
- VS Code Python/Jupyter extensions
