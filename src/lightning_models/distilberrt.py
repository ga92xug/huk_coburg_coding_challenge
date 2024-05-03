import pytorch_lightning as pl
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torchmetrics import F1Score
from torch.optim import AdamW

from src.dataloader import clean_text

class distilbert(pl.LightningModule):
    def __init__(self, num_labels: int, lr: float = 5e-5):
        super().__init__()
        self.lr = lr
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels)
        self.f1_metric = F1Score(task="multiclass", num_classes=num_labels, average='macro')

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
        return loss

    def training_step(self, batch, batch_idx):
        output = self.forward(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self.inference_step(batch, 'test')

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        opt = {
            "optimizer": optimizer,
        }
        return opt


if __name__ == '__main__':
    model = distilbert(4)
    print(model)