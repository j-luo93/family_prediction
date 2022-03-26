from argparse import ArgumentParser
from collections import Counter
from os import name
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import BATCH_SIZE, HS1, HS2, INPUT_SIZE, KERNEL_SIZE, NUM_CLASSES
from data import CombinedDataset, FamilyDataset, OneLangDataset, get_dataset


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(INPUT_SIZE, HS1, KERNEL_SIZE), nn.ReLU())
        self.dense1 = nn.Sequential(nn.Linear(HS1, HS2), nn.Tanh())
        self.dense2 = nn.Sequential(nn.Linear(HS2, NUM_CLASSES))

        self.lang = nn.Parameter(torch.Tensor(HS1, HS1))
        nn.init.uniform_(self.lang, -0.05, 0.05)
        self.Wa = nn.Parameter(torch.Tensor(HS1, HS1))
        nn.init.uniform_(self.Wa, -0.05, 0.05)
        nn.init.uniform_(self.dense1[0].weight, -0.05, 0.05)
        nn.init.uniform_(self.dense2[0].weight, -0.05, 0.05)
        nn.init.xavier_normal_(self.conv[0].weight)

    def forward(self, inputs):
        bs, ns, hs, ml = inputs.shape
        x = self.conv(inputs.view(bs * ns, hs, ml))
        x = x.max(dim=2)[0].view(bs, ns, -1)
        score = x @ self.Wa
        score = score @ self.lang
        weights = score.log_softmax(dim=1).exp()
        # temp_weights = tf.expand_dims(weights, axis = -1)
        # temp_x = tf.math.reduce_sum(inputs * temp_weights, axis=1)
        # temp_x = tf.math.reduce_mean(temp_x, axis=0)
        # tf.print(temp_x)
        x = (x * weights).sum(dim=1)

        x = self.dense1(x)
        x = self.dense2(x)
        return x


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data_save_path', help='Path to save the loaded dataset.')
    parser.add_argument('model_save_path', help='Path to save the train model.')
    args = parser.parse_args()

    mdl = MyModel()
    data_path = Path(args.data_save_path)
    if data_path.exists():
        comb_dataset = torch.load(data_path)
    else:
        _, comb_dataset = get_dataset()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(comb_dataset, data_path)

    # This is for training.
    # comb_dl = DataLoader(comb_dataset, shuffle=True, batch_size=batch_size, drop_last=True)  # a data loader is a dynamic loader function from the dataset (randomly)
    # This is for evaluation -- therefore no shuffling and do not drop last.

    # rom_dl = DataLoader(rom_dataset, shuffle=False, batch_size=batch_size, drop_last=False)
    # germ_dl = DataLoader(germ_dataset, shuffle=False, batch_size=batch_size, drop_last=False)

    # deu_dl = DataLoader(deu_dataset, shuffle=False, batch_size=batch_size, drop_last=False)
    training_dl = DataLoader(comb_dataset, shuffle=True, batch_size=BATCH_SIZE, drop_last=False)
    # weights = torch.ones(2000)

    # weights[1000:2000] = 2

    # weighted_sampler = WeightedRandomSampler(weights, batch_size)
    # weighted_rom_dl = DataLoader(rom_dataset, sampler=weighted_sampler, drop_last=True)
    # rom_dl = DataLoader(rom_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    # germ_dl = DataLoader(germ_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    # train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)  # gets passed the function to be optimized

    for epoch_i in range(30):
        print('Epoch: ', epoch_i)
        agg_loss = 0.0  # loss initialization
        total_n_batches = 0  # init number of batches
        for batch in training_dl:
            mdl.train()  # set flag for model to be trained
            optimizer.zero_grad()  # zero out all gradients

            logits = mdl(batch['data'])
            loss = loss_func(logits, batch['label'])

            loss.backward()  # runs backpropagation - optimizer will collect gradients automatically
            optimizer.step()  # optimizer takes one gradient step

            bs = len(batch['data'])
            agg_loss += loss.item() * bs  # bc loss is averaged - we have to mulitply
            total_n_batches += bs
        mean_loss = agg_loss / total_n_batches
        print('mean train loss: ' + f'{mean_loss:.6f}')

        with torch.no_grad():  # this is evaluation
            """
            dls_to_eval = {
                'rom': rom_dl,
                'germ': germ_dl
            }
            """

            # for name, dl in dls_to_eval.items():
            n_correct = 0.  # this is for computing accuracy
            n_total = 0
            allpreds = []
            for batch in training_dl:
                mdl.eval()  # set flag for model to be evaluated

                logits = mdl(batch['data'])
                predicted = logits.max(dim=-1)[1]  # predicted family index

                correct = predicted == batch['label']
                n_correct += correct.sum()
                n_total += len(correct)
                allpreds.append(predicted.tolist())

            mean_acc = n_correct / n_total
            print(f'mean accuracy: ' + f'{mean_acc:.3f}')

        Path(args.model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(mdl.state_dict(), args.model_save_path)
        # eval dataset must contain all 'deu' and a sample from the rest
