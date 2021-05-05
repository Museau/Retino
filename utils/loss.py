import copy

import torch
import torch.nn.functional as F

from torch import nn


def get_loss(cfg_loss):
    """
    Build the loss with the proper parameters and return it.

    Parameters
    ----------
    cfg_loss : dict
        Dictionary containing the name of the loss to use and it's specific configs.

    Returns
    -------
    loss_function : function
        The loss function.
    """
    loss_args = copy.deepcopy(cfg_loss)

    # Import proper loss class
    if loss_args['name'] == 'BinaryFocalLoss':
        exec(f"from utils.loss import {loss_args['name']}")
    else:
        exec(f"from torch.nn import {loss_args['name']}")
    loss_class = eval(loss_args['name'])
    del loss_args['name']

    # Convert to torch.tensor some argument that requires it.
    for arg_name in ['pos_weight', 'weight']:
        if arg_name in loss_args:
            loss_args[arg_name] = torch.tensor(loss_args[arg_name])

    loss_function = loss_class(**loss_args)

    return loss_function


class BinaryFocalLoss(nn.modules.loss._Loss):
    """Criterion that computes Focal loss.
    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=.25, gamma=2, reduction='mean'):
        """Criterion that computes Focal loss.

        Parameters
        ----------
        alpha : float
            Weighting factor between 0 and 1.
            Default: 0.25
        gamma : float
            Focusing parameter >= 0.
            Default: 2
        reduction : str
            Specifies the reduction in  [none, mean, sum], to apply to the output.
            Default: mean
        """
        super(BinaryFocalLoss, self).__init__()
        alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma
        self.reduction = reduction

        self.register_buffer('alpha', alpha)

    def __str__(self):
        return f"BinaryFocalLoss (\n\talpha: {self.alpha}\n\tgamma: {self.gamma}\n\treduction: {self.reduction}\n)"

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt)**self.gamma * BCE_loss

        # Apply reduction
        if self.reduction == 'none':
            loss = F_loss
        elif self.reduction == 'mean':
            loss = torch.mean(F_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(F_loss)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")
        return loss
