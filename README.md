# Smart Transportation Intelligence

## Implementation Modules

1. Data cleaning
2. Exploratory data analysis
3. Feature engineering
4. Machine learning model training
5. Explainable AI using SHAP and LIME
6. Streamlit dashboard

## Setup

```bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run in Order

```bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/eda.py
python src/model_training.py
python src/explainability.py
streamlit run app.py
```

## Important Note

The accident risk target is an academic proxy label created from 2023 state accident counts. It is not a ground-truth accident-event label and should not be described as a real-time prediction system.
