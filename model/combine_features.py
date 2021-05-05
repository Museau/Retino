import torch
import torch.nn as nn


class CombineFeatures(nn.Module):

    def __init__(self, base_net, combine_both_eye, drop_rate, num_classes, grad_checkpointing=False):
        """
        Correspond to the combine features from multi-view module.

       Parameters
        ----------
        base_net : nn.Module
            Instance network with a forward method
            implemented.
        combine_both_eye : bool
            Use both eye for prediction.
        drop_rate : float
            Dropout rate.
        num_classes : int
            Number of classes.
        """
        super(CombineFeatures, self).__init__()

        self.base_net = base_net
        self.combine_both_eye = combine_both_eye

        self.in_dim = self.base_net.emb_size
        if self.combine_both_eye:
            self.in_dim = self.in_dim * 2

        classifier = []
        if drop_rate > 0.:
            classifier += [nn.Dropout(drop_rate)]

        classifier += [nn.Linear(self.in_dim, num_classes)]
        self.fc_layers = nn.Sequential(*classifier)

        if num_classes == 1:

            def get_proba(logits):
                return torch.sigmoid(logits)
        else:

            def get_proba(logits):
                m = torch.nn.Softmax(dim=1)
                return m(logits)

        self.final_activation = get_proba

        self.grad_checkpointing = grad_checkpointing

    def forward(self, eyes_img, eyes_mask):
        """
        Forward path.

        Parameters
        ----------
        eyes_img : list of ndarray inputs
            Inputs to the network.
        eyes_mask : ndarray
            ndarray indicating in the case of each views if the element of the list of
            ndarray corresponds to a real input or a zero padding.

        Returns
        -------
        logits : ndarray
            Returns logits pre final activation of the network.
        """
        eyes_emb = []
        for side_id in range(len(eyes_img)):
            side_emb = []
            for view_id, eye_views in enumerate(eyes_img[side_id]):
                if self.grad_checkpointing:
                    emb = torch.utils.checkpoint.checkpoint(self.base_net, eye_views)
                else:
                    emb = self.base_net(eye_views)
                # Tweak for Inception/GoogLeNet output
                # They return named tuples containing the tensor instead of returning the tensor directly.
                if not torch.is_tensor(emb):
                    emb = emb[0]

                mask = eyes_mask[side_id][:, view_id].view(-1, 1).expand_as(emb)
                emb = emb * mask
                side_emb.append(emb)

            n_views = torch.sum(eyes_mask[side_id], 1).view(-1, 1)
            sum_emb = torch.sum(torch.stack(side_emb), 0)
            eyes_emb.append(torch.div(sum_emb, n_views))

        embedding = torch.cat(eyes_emb, 1)

        logits = self.fc_layers(embedding)
        return logits
