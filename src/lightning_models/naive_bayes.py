import pickle
import pytorch_lightning as pl

from src.dataloader import clean_text


class naive_bayes(pl.LightningModule):
    def __init__(self):
        super().__init__()
        with open('saved_models/naive_bayes.pkl', 'rb') as file: 
            self.model = pickle.load(file) 

    def predict(self, text):
        pre_processing_text = clean_text(text)
        return self.model.predict([pre_processing_text])[0]