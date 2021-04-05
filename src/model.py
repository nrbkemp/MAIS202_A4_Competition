import pytorch_lightning as pl
from utils import vgg_conv_block
import torch as pt
import torch.nn as tnn
from torch.nn import functional as F


class MNISTClassifier(pl.LightningModule):

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.layer1 = vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block(
            [64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        # input for layer 5 is the dim of flattened form of output form layer 4
        self.layer5 = vgg_fc_layer(16384, 4096)
        self.layer6 = vgg_fc_layer(4096, 4096)

        # Final layer
        self.layer7 = tnn.Linear(4096, 10)  # number of classes = 10

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        vgg16_features = self.layer4(x)  # end of VGG blocks

        x = vgg16_features.view(x.size(0), -1)  # Flatten the features
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = pt.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate)
        return optimizer
