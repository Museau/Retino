import torch.nn as nn

from collections import OrderedDict

from utils.model import build_norm_layer, IdentityLayer

BUILDS = {"cnn3layers1conv": [32, 'M', 64, 'M', 128],
          "cnn3layers2conv": [32, 32, 'M', 64, 64, 'M', 128, 128],
          "cnn4layers1conv": [32, 'M', 64, 'M', 128, 'M', 128],
          "cnn4layers2conv": [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]}


class SimpleNet(nn.Module):
    """
    VGG inspired CNN without the head.
    """

    def __init__(self, cfg, in_channels=3):
        """
        Parameters
        ----------
        cfg : dict
            name : str
                Model name.
            norm_layer : dict, optional
                name : str
                    Name of the normalization layer
                .... : obj
                    Other entry the represents the parameters of the given norm_layer
        in_channels : int (optinal, default: 3)
            Number of input channels.
        """
        super(SimpleNet, self).__init__()
        layers_def = BUILDS[cfg['name']]

        self.in_channels = in_channels
        self.emb_size = layers_def[-1]

        # Prepare norm layer
        self._norm_layer = build_norm_layer(cfg.get('norm_layer', None))

        self.net = self._make_layers(layers_def)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, self.emb_size)
        return output

    def _make_layers(self, layers_def):
        pool_count = 0
        conv_count = 0
        bn_count = 0
        layers = []
        in_channels = self.in_channels
        for layer_def in layers_def:
            if layer_def == 'M':
                pool_count += 1
                layers += [(f'maxpool{pool_count}', nn.MaxPool2d(kernel_size=2))]
            else:
                nb_features = layer_def
                conv_count += 1
                layers += [(f'conv{conv_count}', nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=nb_features, stride=1, padding=1))]

                # Add normalization layer if necessary
                if self._norm_layer is not IdentityLayer:
                    bn_count += 1
                    layers += [(f'bn{bn_count}', self._norm_layer(nb_features))]

                layers += [(f'relu{conv_count}', nn.ReLU())]
                in_channels = nb_features
        layers += [('avgpool', nn.AdaptiveAvgPool2d((1, 1)))]
        return nn.Sequential(OrderedDict(layers))
