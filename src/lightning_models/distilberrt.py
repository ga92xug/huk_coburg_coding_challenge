import pytorch_lightning as pl
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassConfusionMatrix
from torch.optim import AdamW

from src.dataloader import clean_text

class Distil_Bert(pl.LightningModule):
    def __init__(self, num_labels: int, lr: float = 5e-5):
        super().__init__()
        self.lr = lr
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.f1_metric = F1Score(task="multiclass", num_classes=num_labels, average='macro')
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_labels, normalize='true')

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        return output
    
    def predict(self, text: str):
        if not hasattr(self, 'tokenizer'):
            # Load the tokenizer if it hasn't been loaded yet
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        pre_processing_text = clean_text(text)
        inputs = self.tokenizer(pre_processing_text, return_tensors='pt')
        output = self.forward(**inputs)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)
        return preds.item()

    def inference_step(self, batch, mode):
        output = self.forward(**batch)
        loss = output.loss
        self.log(f'{mode}_loss', loss, prog_bar=True)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)
        labels = batch['labels']
        f1 = self.f1_metric(preds, labels)
        self.log(f'{mode}_f1', f1, prog_bar=True)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.inference_step(batch, 'val')
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.inference_step(batch, 'test')
        self.confusion_matrix(preds, labels)
        return loss
    
    def on_test_epoch_end(self):
        fig_, ax_ = self.confusion_matrix.plot(labels=['Irrelevant', 'Negative', 'Neutral', 'Positive'])
        # save figure normal 
        fig_.savefig('figures/confusion_matrix_distil_bert.png')

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        opt = {
            "optimizer": optimizer,
        }
        return opt


if __name__ == '__main__':
    model = Distil_Bert(4)
    print(model)