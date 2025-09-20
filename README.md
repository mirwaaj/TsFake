# Fake News Detection with Machine Learning

## Overview
This project investigates the use of machine learning techniques to classify news articles as real or fake. We implement a baseline model using **TF-IDF vectorization** and a **Passive Aggressive Classifier**, and evaluate its performance on a benchmark dataset. The aim is to provide a reproducible foundation for future exploration of misinformation detection methods.

## Objectives
- Develop a reproducible pipeline for fake news detection.  
- Benchmark a lightweight model as a baseline for future comparison.  
- Evaluate performance using accuracy, precision, recall, and confusion matrices.  
- Prepare the framework for extensions such as deep learning models or evidence-based verification.


## Dataset
The project uses the **Kaggle Fake News dataset**:  
https://www.kaggle.com/c/fake-news  

Steps to obtain:  
1. Create a Kaggle account and accept competition rules.  
2. Download `train.csv` and place it in the `data/` directory.  
3. Do not commit large dataset files to the repository.

## Methodology
1. **Preprocessing**  
   - Remove missing values and normalize text.  
   - Split dataset into training and testing sets.  

2. **Feature Extraction**  
   - Apply TF-IDF vectorization with English stop words removed.  
   - Limit features by maximum document frequency to reduce noise.  

3. **Model Training**  
   - Use a Passive Aggressive Classifier with 50 iterations.  
   - Train on TF-IDF features of the training set.  

4. **Evaluation**  
   - Compute accuracy, precision, recall, and F1 score.  
   - Generate confusion matrices for error analysis.  

## Results
- Baseline model achieves accuracy in the range of 90–94% depending on dataset split.  
- Confusion matrix highlights differences in false positives vs. false negatives.  
- Provides a benchmark for more advanced approaches (e.g., transformer models).  

Detailed results and plots are stored in the `results/` directory.  

## Usage
### 1. Install dependencies
```bash
pip install numpy pandas scikit-learn
pip install -r requirements.txt
```
### 2. Train the model
Open and run the baseline notebook:
```bash 
notebooks/model_baseline.ipynb
```
### 3. Make predictions
Example utility function (src/utils.py):
```bash
streamlit run app/app.py
```
### 4. Run demo appication
```bash 
streamlit run app/app.py
```
