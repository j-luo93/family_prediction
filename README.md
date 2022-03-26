# Install
Run `pip install -r requirements` to install the dependencies.

# Run
* First, make sure you have three files in the `data` folder:
    * `phoibledata.npy`
    * `northeuralex-0.9-language-data.tsv`
    * `northeuralex-0.9-forms.tsv`
* Run `python train.py output/dataset.pth output/model.pth` to train the model. It will save two `pth` files in the `output` folder: the dataset `dataset.pth` and the trained model `model.pth`.
* Run `python ig.py output/dataset.pth output/model.pth output/attributions.pth` to apply the integrated gradients method to obtain the feature attribution scores on a random batch. The output
    is saved in `output/attributions.pth`.