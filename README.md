# Sequence-to-Sequence Transformer Model for Time Series Prediction

This project implements a sequence-to-sequence transformer model based on the "Attention is All You Need" architecture for time series prediction using the ETTm1 dataset. The model processes a 6-dimensional input sequence to predict a 1-dimensional output sequence.

# Installation
```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

# Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```
# Running the Project

### Preprocesses the dataset and saves the processed data
We use the ETTm1 dataset, a public dataset for electricity transformer temperature.
```
python data_preprocessing.py
```

### Train the model using the processed data

```
python train.py
```
### Evaluates the trained model on the validation set
```
python evaluate.py
```

