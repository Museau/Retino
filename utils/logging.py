import os
import tempfile

from collections import OrderedDict

import mlflow
import numpy as np
import ruamel.yaml as yaml
import matplotlib.pyplot as plt

from torchvision.utils import save_image, make_grid


def print_and_log_scores(scores, epoch=None, log=True):
    """
    Print and log to MLFlow the scores of a given epoch.

    Parameters
    ----------
    scores : dict
        Dictionary of scores for the given epoch.
    epoch : int
        The epoch number. Default: None.
    log : bool
        Whether or not to log values to mlflow. Default: True.
    """
    for s in scores:
        formated_value = f"\t{s.replace('_', ' ').title()}: "
        # Log scores to mlflow and print them
        if s.endswith("confusion_matrix"):
            if log:
                # For the confusion matrix the true positive for each class
                # and the errors are logged
                mode = s.replace("confusion_matrix", "")
                num_class = len(scores[s][0])
                for i in range(num_class):
                    for j in range(num_class):
                        if i == j:
                            score_name = f"{mode}class{i}_tp"
                        else:
                            score_name = f"{mode}class{i}_err{j}"
                        mlflow.log_metric(score_name, scores[s][i, j], step=epoch)

            formated_value += "".join([f"\n\t {r}" for r in scores[s]])

        elif s.endswith("per_class"):
            if log:
                # Log precision, recall, f1 score and support per class
                for idx, class_s in enumerate(scores[s]):
                    mlflow.log_metric(f"{mode}class{idx}_{s.split('_')[1]}", class_s, step=epoch)
            formated_value += f"\n\t {scores[s]}"

        else:
            if log:
                mlflow.log_metric(f"{s}", scores[s], step=epoch)
            formated_value += f"{scores[s]}"

        print(formated_value)


class GradientLogger:
    """
    Logger class that keep track of gradients and log different stats over training.
    """

    def __init__(self, model, log_every_n_epoch=0):
        """
        Parameters
        ----------
        model : nn.Module
            The PyTorch model to log info about its gradients.
        log_every_n_epoch : int [optional]
            When to log information about the gradients.
            Setting it to 0 disable the logging. Default: 0
        """

        self._epoch = 0

        self._params = OrderedDict()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                self._params[param_name] = param

        self._mlflow_epoch_folder = f"gradients/epoch_{self._epoch}"
        self._tmp_dir = tempfile.TemporaryDirectory()
        self._flow_img_path = os.path.join(self._tmp_dir.name, "global_flow.png")
        self._log_every_n_epoch = log_every_n_epoch

    def should_log(self):
        """
        Determine if the current epoch is one to be logged.
        """
        if self._log_every_n_epoch > 0:
            return ((self._epoch - 1) % self._log_every_n_epoch) == 0
        return False

    def _log_histogram(self, layer_id, layer_name, grad, stats):
        """
        Log the values of the absolute gradient of a given layer in histogram form.
        Keep track of min, max, mean and L2 norm of the gradients in the title of the plot.
        Log the resulting histogram as an image artifact to MLflow.

        Parameters
        ----------
        layer_id : int
            ID to determine the order of the layers. Only used in the filename of the hist.
        layer_name : str
            Name of the current layer.
        grad : torch.Tensor
            The gradients of the current layer.
        stats : dict
            Dictionary containing the min, max, mean and L2 norm of the gradient of the curent layer.

        """

        # Using the Struge formula as the one used by auto is slow and
        # Can give 40000+ bins and the hist looks empty
        nb_bin = np.ceil(1 + np.log2(stats['count']))

        plt.title(f"Epoch {self._epoch}\nMin:{stats['min']} | Max:{stats['max']} | Mean:{stats['mean']} | L2:{stats['norm_str']}")
        plt.hist(grad, bins=int(nb_bin), label=f"{layer_name} layer abs grad")
        plt.legend()

        image_path = os.path.join(self._tmp_dir.name, f"{layer_id:02d}_{layer_name}.png")
        plt.savefig(image_path)
        plt.clf()

        mlflow.log_artifact(image_path, artifact_path=f"{self._mlflow_epoch_folder}/layers")

    def _compute_layer_grad_stats(self, param):
        grad = param.grad.abs().cpu().numpy().flatten()
        norm = param.grad.norm(2).cpu().numpy()

        stats = {'count': len(grad)}
        stats['min'] = np.format_float_scientific(np.min(grad), precision=4)
        stats['max'] = np.format_float_scientific(np.max(grad), precision=4)
        stats['mean'] = np.format_float_scientific(np.mean(grad), precision=4)
        stats['norm'] = norm
        stats['norm_str'] = np.format_float_scientific(norm, precision=4)

        return grad, stats

    def log_full(self):
        """
        Generate the histogram of absolute gradients per layer of given model on a given batch.
        Keep track of min, max, mean and L2 norm of the gradients.

        Update a graph of the flow of gradients throughout all layers of a model for a given batch.

        Log the histogram images as artifact in MLflow.
        """

        grads = np.array([])
        norm_by_layer = OrderedDict()
        for i, param_name in enumerate(self._params.keys()):
            p = self._params[param_name]
            g, stats = self._compute_layer_grad_stats(p)
            grads = np.append(grads, g)
            norm_by_layer[param_name] = stats['norm']

            self._log_histogram(i, param_name, g, stats)

        mlflow.log_metric("model_abs_gradient_min", np.min(grads), step=self._epoch)
        mlflow.log_metric("model_abs_gradient_max", np.max(grads), step=self._epoch)
        mlflow.log_metric("model_abs_gradient_mean", np.mean(grads), step=self._epoch)
        mlflow.log_metric("model_global_l2_norm", np.linalg.norm(grads, 2), step=self._epoch)

        self._reset_plot()
        self._plot_grad_flow(norm_by_layer)

    def log(self):
        """
        Update a graph of the flow of gradients throughout all layers of a model for a given batch.
        """
        norm_by_layer = OrderedDict()
        for i, param_name in enumerate(self._params.keys()):
            p = self._params[param_name]
            g, stats = self._compute_layer_grad_stats(p)
            norm_by_layer[param_name] = stats['norm']

        self._plot_grad_flow(norm_by_layer)

    def _plot_grad_flow(self, norm_by_layer):
        plt.plot(list(norm_by_layer.values()), alpha=0.3, color="b")

    def _reset_plot(self):
        plt.clf()
        nb_layers = len(self._params)
        plt.hlines(0, 0, nb_layers + 1, linewidth=1, color="k")
        plt.xticks(range(0, nb_layers, 1), self._params.keys(), rotation="vertical")
        plt.xlim(xmin=0, xmax=nb_layers)
        plt.xlabel("Layers")
        plt.ylabel("l2-norm gradient")
        plt.title("Gradient flow")
        plt.grid(True)

    def end_epoch(self):
        """
        If this epoch should be logged (see should_log function),
        save a graph of the flow of gradients throughout all layers of a model for a a whole epoch
        and increment the current epoch.
        """
        if self.should_log():
            plt.savefig(self._flow_img_path, bbox_inches='tight')
            self._reset_plot()

            mlflow.log_artifact(self._flow_img_path, artifact_path=self._mlflow_epoch_folder)
        self._epoch += 1
        self._mlflow_epoch_folder = f"gradients/epoch_{self._epoch}"

    def __del__(self):
        self._tmp_dir.cleanup()


def _rescale_0_1(batch):
    """
    Rescale all image from batch, per channel, between 0 and 1
    """
    for image_id in range(batch.size(0)):
        for channel_id in range(batch[image_id].size(0)):
            pix_min = batch[image_id][channel_id].min()
            pix_range = batch[image_id][channel_id].max() - pix_min
            batch[image_id][channel_id].sub_(pix_min).div_(pix_range)
    return batch


def log_batch_images(batch, epoch):
    """
    Take a batch of images, rescale them between 0 and 1,
    align them in a grid which is as a square as possible an
    log the resulting image to MLFlow as an artifact.

    Parameters
    ----------
    batch : torch.Tensor
        Tensor list of images.
    epoch : int
        Epoch ID of the batch to log.
    """
    nrow = int(np.ceil(np.sqrt(batch.size(0))))
    grid = make_grid(_rescale_0_1(batch), nrow=nrow)

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, f"first_batch_of_epoch_{epoch}.png")
        save_image(grid, image_path)
        mlflow.log_artifact(image_path, artifact_path="image_samples")


def log_dict_to_artefact(data, artifact_name, artifact_path):
    """
    Save dictionary as an artifact.

    Parameters
    ----------
    data : dict
        Dictionary to be saved as an artifact.
    artifact_name : str
        Name of the artifact file in MLFlow
    artifact_path : str
        Where to place the file in the virtual folder structure of MLFlow.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path_local = os.path.join(tmp_dir, artifact_name)
        with open(artifact_path_local, 'w') as f:
            yaml.dump(data, f, Dumper=yaml.RoundTripDumper)

        mlflow.log_artifact(artifact_path_local, artifact_path=artifact_path)


def log_config(cfg, prefix=''):
    """
    Log recursively every config in cfg as an MLFlow param.

    Ex:
    optimizer:
      weight_decay:
        rate: 0.001
    Will be logged as `optimizer.weight_decay.rate : 0.001`.

    Parameters
    ----------
    cfg : dict
        A dictionary containing all configs.
    prefix : str
        A prefix to append in front of the name of the key before logging it.
    """
    for conf_name in cfg.keys():
        if type(cfg[conf_name]) is dict:
            log_config(cfg[conf_name], prefix=f"{prefix}{conf_name}.")
        else:
            mlflow.log_param(f"{prefix}{conf_name}", cfg[conf_name])
