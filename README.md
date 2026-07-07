# Fashion-MNIST Classifier with PyTorch + Optuna

A simple artificial neural network (ANN) built in PyTorch to classify clothing images from the **Fashion-MNIST** dataset, with automatic hyperparameter tuning using **Optuna**.

## What this project does

- Loads the Fashion-MNIST training data (`fashion-mnist_train.csv`)
- Visualizes a few sample images with their labels
- Normalizes pixel values and splits the data into train/test sets
- Wraps the data in a custom PyTorch `Dataset` class
- Defines a fully-connected neural network with configurable depth, width, dropout, and batch normalization
- Uses **Optuna** to automatically search for the best combination of hyperparameters (number of layers, neurons per layer, learning rate, batch size, optimizer, dropout rate, weight decay, epochs)
- Reports the best accuracy and best hyperparameter set found

## Tech stack

- Python
- PyTorch
- Pandas
- Scikit-learn (train/test split)
- Matplotlib (visualization)
- Optuna (hyperparameter optimization)
- Google Colab (originally built/run here, with GPU/T4 support)

## Dataset

This notebook expects the **Fashion-MNIST** CSV dataset (`fashion-mnist_train.csv`), where:
- Column `label` → clothing category (0–9)
- Columns `pixel1` ... `pixel784` → flattened 28x28 grayscale pixel values

You can download it from [Kaggle - Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) and place it in the same folder as the notebook (or update the file path in the code).

## How to run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install the required packages:
   ```bash
   pip install pandas scikit-learn torch matplotlib optuna
   ```

3. Make sure the dataset file `fashion-mnist_train.csv` is available at the path used in the notebook (update it if needed).

4. Open the notebook and run all cells:
   ```bash
   jupyter notebook bulid_ANN.ipynb
   ```

5. The Optuna study will run 10 trials by default, trying different hyperparameter combinations, and print the accuracy for each trial. At the end, you'll see:
   - `study.best_value` → best accuracy achieved
   - `study.best_params` → the hyperparameter combination that achieved it

## Notes

- Training uses GPU automatically if available (`cuda`), otherwise falls back to CPU.
- Random seed is fixed (`torch.manual_seed(42)`) for reproducibility.
- You can increase `n_trials` in `study.optimize(objective, n_trials=10)` to search more thoroughly (this will take longer to run).

## Possible improvements

- Save the best model after tuning instead of just printing the best params
- Add a proper validation set (currently test set is reused for evaluation during tuning)
- Add training/loss curve plots
- Try a CNN instead of a plain ANN for better accuracy on image data

## License

Feel free to use, modify, and share this project.
