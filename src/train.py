from argparse import ArgumentParser
from pytorch_lightning import Trainer

from lightning_model import SentimentClassifier
from dataloader import get_dataloaders

def main(args):
    dataloaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)

    lightning_model = SentimentClassifier(num_labels=4, lr=args.lr)

    trainer = Trainer(max_epochs=args.epochs)
    trainer.fit(lightning_model, dataloaders['train'], dataloaders['valid'])

    trainer.test(lightning_model, dataloaders['test'])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the dataloaders')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloaders')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
