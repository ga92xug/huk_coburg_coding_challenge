from argparse import ArgumentParser
from pytorch_lightning import Trainer
import os
import sys
sys.path.append(os.getcwd())

from lightning_models.distilberrt import Distil_Bert
from dataloader import get_dataloaders

def main(args: ArgumentParser) -> None:
    dataloaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    trainer = Trainer(max_epochs=args.epochs)

    if args.mode == 'train':
        lightning_model = Distil_Bert(num_labels=4, lr=args.lr)
        trainer.fit(lightning_model, dataloaders['train'], dataloaders['valid'])
    elif args.mode == 'load':
        lightning_model = Distil_Bert.load_from_checkpoint('saved_models/distil_bert.pkl', num_labels=4)
        lightning_model.eval()
    else:
        raise ValueError('Invalid mode. Please choose either train or load')

    trainer.test(lightning_model, dataloaders['test'])


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for the dataloaders')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloaders')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--mode', type=str, default='train', help='train or load')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
