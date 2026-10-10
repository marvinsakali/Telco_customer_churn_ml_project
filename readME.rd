# Telco Customer Churn Prediction

A machine learning project for predicting whether a telecom customer is likely to churn. The repository includes exploratory data analysis, data cleaning, preprocessing, and model training workflows for churn prediction.

## Overview

Customer churn is a major business challenge in the telecom industry. This project analyzes customer usage, billing, service plan, and account history to predict churn risk. The goal is to help business teams identify customers likely to leave and take preventive action.

## Business Problem

Telecom providers want to reduce customer attrition by understanding which customers are at risk of cancellation. With churn data, we can build a predictive model to estimate the likelihood that a customer will leave based on their behavior and account profile.

## Dataset

The project uses the public Telco Customer Churn dataset:

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Source: telecom customer churn dataset commonly used in machine learning examples
- Target column: `Churn`

Typical features include:

- customer tenure
- contract type
- monthly charges
- total charges
- internet service type
- payment method
- support services and add-ons

## Repository Structure

```text
Telco_customer_churn_ml_project/
├── Data/
│   ├── raw/
│   └── processed/
├── Notebooks/
│   ├── EDA.ipynb
│   └── data_cleaning.ipynb
├── src/
│   ├── data_processing.py
│   ├── data_utils.py
│   ├── train.py
│   └── ...
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── readME.rd
├── README.md
└── .gitignore
```

## Project Workflow

1. Explore the dataset in the EDA notebook.
2. Clean and prepare the dataset in the data cleaning notebook.
3. Use preprocessing utilities in `src/data_processing.py` and `src/data_utils.py`.
4. Train machine learning models in `src/train.py`.
5. Evaluate model performance and adjust features or models as needed.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- Jupyter Notebook
- Matplotlib / Seaborn (commonly used for EDA and visualization)

## Setup

Create a virtual environment and install the required libraries:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

## Running the Project

Open the notebooks in the `Notebooks/` directory to explore and clean the dataset:

```bash
jupyter notebook
```

Then open:

- `Notebooks/EDA.ipynb`
- `Notebooks/data_cleaning.ipynb`

You can also run training logic from the scripts in the `src/` directory.

## Notes

- The project stores raw and processed data separately under `Data/raw` and `Data/processed`.
- Some scripts currently contain local path references and may need to be adapted to your environment before running.
- This is a practical churn prediction project for learning and experimentation in data science and machine learning.

## Future Improvements

- add model comparison and evaluation metrics
- tune hyperparameters using cross-validation
- build a more robust preprocessing pipeline
- deploy the final model as an API or dashboard
- generate customer-level churn explanations using SHAP or feature importance

## Conclusion

This project demonstrates a complete end-to-end workflow for a telecom churn prediction problem, from data analysis and preprocessing to model training and evaluation. It is suitable for educational use, portfolio work, or as a starting point for a more advanced predictive analytics solution.
