from argparse import ArgumentParser

import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

from config import BATCH_SIZE
from data import get_dataset
from train import MyModel

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_path', help="Path to the saved dataset.")
    parser.add_argument('model_path', help="Path to the saved model.")
    parser.add_argument('output_path', help="Path to save the predicted feature attribution matrices.")
    args = parser.parse_args()

    model = MyModel()
    model.load_state_dict(torch.load(args.model_path))
    comb_dataset = torch.load(args.data_path)
    training_dl = DataLoader(comb_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)

    ig = IntegratedGradients(model)
    # Here we assume that we are using zero input vectors as the baselines.
    batch = next(iter(training_dl))
    # Right now, the attributed on computed based on raw logits, not probs.
    attributions, approximation_error = ig.attribute(batch['data'],
                                                     target=4,
                                                     method='gausslegendre',
                                                     return_convergence_delta=True)
    torch.save(attributions, 'attributions.pth')
