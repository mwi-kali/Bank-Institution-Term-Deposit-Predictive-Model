# Bank Institution Term-Deposit Subscription Prediction

This repository contains a complete end-to-end pipeline for exploring, feature-engineering, modeling, and deploying predictive models to forecast whether a client will subscribe to a bank term deposit. It leverages both the original UCI **Bank Marketing** dataset and an enriched version with national economic indicators.

---

## Project Structure

```
.
├── data/
│   ├── bank-full.csv
│   ├── bank-names.txt
│   ├── bank-additional-full.csv
│   ├── bank-additional-names.txt
│   ├── feature_names.csv
│   ├── X_train_preprocessed.csv
│   ├── X_test_preprocessed.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── notebooks/
│   ├── Bank_Institution_Term_Deposit_Predictive_Model_Exploratory_Data_Analysis_(EDA).ipynb      
│   ├── Bank_Institution_Term_Deposit_Predictive_Model_Feature_Engineering.ipynb      
│   └── Bank_Institution_Term_Deposit_Predictive_Model_Evaluation.ipynb       
│
├── src/
│   ├── __init__.py
│   ├── dashboard.py                     
│   ├── config.py                         
│   ├── data_processing.py 
│   ├── data_loader.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── preprocessing.py
│   └── modeling.py                    
│
├── models/
│   ├── preprocessing_pipeline.pkl     
│   ├── full_pipeline_rf.pkl   
│   ├── feature_pipeline.pkl          
│   ├── stacked_pipeline.pkl          
│   └── final_model_comparison_metrics.csv
│
├── requirements.txt                      # Alphabetical list of Python dependencies
├── README.md                             # Project overview & instructions
└── .gitignore
```

---

## 🔧 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/bank-term-deposit-prediction.git
   cd bank-term-deposit-prediction
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the data**  
   - Place `bank-full.csv` and `bank-additional-full.csv` in the `data/` folder.  
   - (Available from the UCI ML Repository or the [Moro et al. publications](http://hdl.handle.net/1822/14838) & [Decision Support Systems article](http://dx.doi.org/10.1016/j.dss.2014.03.001).)

---

##  Usage

### 1. Run Notebooks
Open the Jupyter notebooks in `notebooks/` for a step-by-step walkthrough:
```bash
jupyter lab
```

### 2. Launch the Dashboard
Run an interactive dashboard combining all phases:
```bash
streamlit run src/dashboard.py
```
Navigate via sidebar tabs That represent the steps:  
  – **Exploratory Data Analysis (EDA)**  
  – **Feature Engineering and Preprocessing**  
  – **Model Tuning**  
  – **Model Evaluation**

