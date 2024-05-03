# Introduction
The repo for the coding challenge from Huk-Coburg. The challenge has 2 parts:

## Part A: NLP
Define metric macro average F1-score, since the labels are not entirely equal distributed. 

### Explorative data analysis
- See notebooks/explore.ipynb for details
- Load and check the data. 
- Transform (clean the texts)
- Naive Bayes as a baseline for sentiment analysis
- F1-score (macro) = 0.78

### Model training
- only use text not product as input
- F1-score (macro) = 0.95

## Part B: ML-Engineering
- Download Distil bert from https://we.tl/t-PHmpxkyO90 and save under saved_models/distil_bert.pkl
- Build and run docker container with: 
docker build -t my-flask-app .
docker run -p 4000:5000 my-flask-app

Can then be tested with
python src/deploy/request_test.py

- missing request api so das man auch programatically requests senden kann




