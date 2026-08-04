# Credit Scoring Model

## Project description

This project focuses on solving a binary classification problem for evaluating the creditworthiness of borrowers. Based on a set of features describing the client and their credit history, a machine learning model is built to predict the probability of default on a loan.

### Main stages:
1. **Exploratory Data Analysis (EDA)** - analyzing distributions, missing values, and correlations
2. **Data preprocessing** - cleaning, feature engineering, filling missing values
3. **Hyperparameter tuning** - optimizing model parameters with Optuna
4. **Building the final model** - training CatBoost with the best parameters
5. **Feature importance analysis** - identifying the key factors influencing the loan approval decision

## Project structure

```
Shift-ML/
├── credit_scoring.ipynb           # Main Jupyter Notebook with the full analysis
├── shift_ml_2026_train.csv        # Training data
├── shift_ml_2026_test.csv         # Test data
├── submission.csv                 # Final predictions
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation (this file)
```

## Installation and running

### Requirements
- Python 3.8+
- Jupyter Notebook
- GPU (optional, to speed up CatBoost training)

### Installing dependencies
```bash
pip install -r requirements.txt
```

Main libraries:
- `pandas` - data processing
- `numpy` - numerical computations
- `scikit-learn` - machine learning tools
- `catboost` - gradient boosting
- `optuna` - hyperparameter optimization
- `matplotlib`, `seaborn` - visualization

### Running the notebook
```bash
jupyter notebook credit_scoring.ipynb
```

### Data
The training and test samples are stored on Google Drive:
https://drive.google.com/drive/folders/1zE12mkEv9r0WQVUWsudMMWrfVTfwDYsu?usp=sharing

## Main components

### 1. Exploratory Data Analysis (EDA)

**Missing value analysis:**
- Features with more than 50% missing values were removed
- Rows with more than 50% missing values were removed

**Correlation analysis:**
- Correlation matrix visualization
- Identification of strongly correlated pairs (|corr| > 0.95)
- Removing one of the correlated features

**Categorical variable analysis:**
- Category imbalance analysis
- Identification of rare categories (frequency = 1)

### 2. Data preprocessing

**Feature selection:**
- Features with > 50% missing values were removed
- Features with category imbalance (one category > 90%) were removed
- Features with high correlation (> 0.95) were removed

**Feature engineering:**
- **Experience (experience_num)**: converting string format to years (< → 0.5, 10+ → 10)
- **Date of the first loan**: extracting the year from the date
- **Borrower profession**:
  - Rare professions (frequency = 1) are encoded as '1'
  - Missing values are encoded as '0'
  - TargetEncoding is applied to convert them to numeric form

**Missing value handling:**
- Numeric missing values: filled with the median depending on the target variable
- Categorical missing values: filled with the mode depending on the target variable

**Data split:**
- Split into training (80%) and validation (20%) sets
- Stratified split by the target variable

### 3. Model building

**CatBoost with parameter optimization (Optuna):**
- Optimization of 9 hyperparameters (iterations, learning_rate, depth, etc.)
- 50 optimization iterations with early stopping
- Optimization metric: ROC-AUC on the validation set

**Optimal parameters of the final model:**
```python
{
    'iterations': 423,
    'learning_rate': 0.11296183714058523,
    'depth': 5,
    'l2_leaf_reg': 2.0718060453463734,
    'random_strength': 5.150015462599627,
    'bagging_temperature': 0.062133050141008464,
    'border_count': 83,
    'grow_policy': 'Depthwise',
    'auto_class_weights': 'SqrtBalanced'
}
```

## Results

### Model quality metrics:
- **ROC-AUC on the training set**: 0.7792
- **ROC-AUC on the validation set**: 0.7615
- **ROC-AUC on the test set (leaderboard)**: ~0.76

### Key features (by importance):
1. **doprating** - the strongest influence on the decision
2. **amount_of_payments_for_overdue_loans** - payment default history
3. **rating** - borrower credit rating
4. **loan_term** - duration of the credit agreement
5. **profession** (TargetEncoding) - professional affiliation
6. **loan_amount** - size of the requested loan

Features with importance < 0.1 have little influence on the model.

## Conclusions

1. **Optimal configuration**: CatBoost shows the best results thanks to effective handling of categorical features and missing value filling.

2. **Model quality**: The model reaches ROC-AUC ~0.76 on test data, indicating good ability to separate default and non-default classes.

3. **Key factors**: Payment history (doprating, overdue payments) and credit rating are the most important factors in loan approval decisions.

4. **Data preprocessing**: Proper handling of missing values (median/mode filling by target) and categorical encoding (TargetEncoding) significantly affect model quality.

## Additional information

- **Data**: The project uses data from a competition on the Shift platform
- **Development platform**: Google Colab using GPU to speed up training
- **Result format**: A CSV file with borrower IDs and predicted default probabilities

## Author
Andrey Dutov

## License

MIT License

Copyright (c) 2026 Andrey Dutov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
