import os
import copy
import shutil
import time

import numpy as np
import torch
import mlflow

from data.data_loader import get_dataloader, update_data_loader_sampler
from utils.checkpoint import save_state_dict, define_train_cfg, set_seed_and_random_states
from utils.logging import print_and_log_scores, log_batch_images, GradientLogger
from utils.scores import compute_scores


def train(data_loaders, model, optimizer, lr_scheduler, loss_function, state, cfg, device):
    """
    Training and validation loop.

    Parameters
    ----------
    data_loaders : dict
                    train : DataLoader
                        Train data loader.
                    valid : DataLoader
                        Validatin data loader.
                    train_exact : DataLoader
                        Dataloader used to compute exact stats on train set.
    model : obj
        The model to use.
    optimizer : obj
        The optimizer to use.
    loss_function : function
        The loss function to use.
    state : dict
        Training state.
    cfg : dict
        Config dictionary.
    device  : str
        Device to use. In ['cpu', 'gpu'].

    Returns
    -------
    float
        Best validation score.

    """

    # Define training config
    patience_metrics = copy.deepcopy(cfg['training']['patience_metrics'])
    train_cfg = define_train_cfg(cfg['training'], patience_metrics, state['train'])

    # Setting raw_labels
    raw_labels = data_loaders['train'].dataset.raw_labels

    # Set grad_logger
    grad_logger = GradientLogger(model, cfg['training']['log_grad_every_n_epoch'])

    # Set under sampling hyper-parameters
    if 'under_sampling' in cfg['dataset']['train']:
        schedule = np.array(cfg['dataset']['train']['under_sampling']['schedule'])
        ratios = np.array(cfg['dataset']['train']['under_sampling']['ratios'])

    if state['random_states'] is not None:
        set_seed_and_random_states(cfg['seed'], state['random_states'], cfg['cuda'])

    # Training
    for epoch in range(train_cfg['starting_epoch'], train_cfg['max_epochs'] + 1):

        start_epoch_time = time.time()

        print(f"\n\n\n#### Epoch {epoch}/{train_cfg['max_epochs']} ####")
        if lr_scheduler:
            print(f"Current LR: {lr_scheduler.get_last_lr()}")

        #
        # Data loader sampler scheduling
        if 'under_sampling' in cfg['dataset']['train'] and epoch in schedule:
            ratio = ratios[np.argwhere(schedule == epoch)[0][0]]
            if ratio is None:
                print("Reset train dataset ratio to original ratio.")
                data_folder = os.path.split(data_loaders['train'].dataset.img_dir)[0]
                data_loaders['train'] = get_dataloader(data_folder, cfg['dataset'], True)['train']
            else:
                data_loaders['train'] = update_data_loader_sampler(data_loaders['train'], ratio)

        #
        # Training loop
        print("\nTraining ...", end=" ")
        start_time = time.time()
        model.train()
        y_true, y_proba, epoch_loss = batch_loop(data_loaders['train'],
                                                 model,
                                                 optimizer,
                                                 loss_function,
                                                 device,
                                                 mode='TRAIN',
                                                 acc_batch_size=cfg['dataset']['train']['accumulated_batch_size'],
                                                 epoch=epoch,
                                                 grad_logger=grad_logger,
                                                 log_image_sample_every_n_epoch=cfg['training']['log_image_sample_every_n_epoch'])

        # Compiling stats
        train_scores = compute_scores(y_true, y_proba, 'TRAIN', raw_labels)
        train_scores['TRAIN_loss'] = round(epoch_loss, 6)

        grad_logger.end_epoch()

        train_time_elapsed = time.time() - start_time
        print(f"Completed in {train_time_elapsed//60:.0f}m {train_time_elapsed%60:.0f}s.")
        print_and_log_scores(train_scores, epoch)

        # Note that if `p_transform` is set to something different than 0
        # This won't be the same exact loss as in the training phase.
        if cfg['training']['exact_train_stats']:
            print("Computing exact TRAIN stats ...", end=" ")
            start_time = time.time()
            model.eval()
            y_true, y_proba, epoch_loss = batch_loop(data_loaders['train_exact'],
                                                     model,
                                                     optimizer,
                                                     loss_function,
                                                     device,
                                                     mode='VALID')

            # Compiling stats
            train_exact_scores = compute_scores(y_true, y_proba, 'TRAIN_exact', raw_labels)
            train_exact_scores['TRAIN_exact_loss'] = round(epoch_loss, 6)

            # Updating the key name that are automatically set to VALID in evaluation mode
            # TODO: Have a smarter evaluation mode
            valid_time_elapsed = time.time() - start_time
            print(f"Completed in {valid_time_elapsed//60:.0f}m {valid_time_elapsed%60:.0f}s.")
            print_and_log_scores(train_exact_scores, epoch)

        #
        # Validation loop
        print("\nValidating ...", end=" ")
        start_time = time.time()
        model.eval()
        y_true, y_proba, epoch_loss = batch_loop(data_loaders['valid'],
                                                 model,
                                                 optimizer,
                                                 loss_function,
                                                 device,
                                                 mode='VALID')

        # Compiling stats
        valid_scores = compute_scores(y_true, y_proba, 'VALID', raw_labels)
        valid_scores['VALID_loss'] = round(epoch_loss, 6)

        valid_time_elapsed = time.time() - start_time
        print(f"Completed in {valid_time_elapsed//60:.0f}m {valid_time_elapsed%60:.0f}s.")
        print_and_log_scores(valid_scores, epoch)

        train_cfg['total_time'] += time.time() - start_epoch_time

        #
        # Learning rate scheduling
        if lr_scheduler:
            lr_scheduler.step()

        #
        # Early stopping and checkpointing best model
        patience_metrics_to_remove = []
        for patience_metric in patience_metrics:
            if valid_scores[f"VALID_{patience_metric}"] > train_cfg[f"best_valid_{patience_metric}"]:

                # Updating training state
                train_cfg[f"patience_{patience_metric}"] = 0
                train_cfg[f"epoch_of_best_valid_{patience_metric}"] = epoch
                train_cfg[f"best_valid_{patience_metric}"] = valid_scores[f"VALID_{patience_metric}"]
                train_cfg[f"loss_for_best_valid_{patience_metric}"] = valid_scores['VALID_loss']

                print(f"\nNew best model for {patience_metric}: checkpointing current state ...", end=" ")
                start_time = time.time()
                # save entire state dict, including model, optimizer and config
                save_fname = os.path.join(cfg['experiment_folder'], f"checkpoint_state_new_{patience_metric}.pth.tar")
                save_state_dict(save_fname,
                                device,
                                model,
                                optimizer,
                                lr_scheduler,
                                train_cfg)
                # To minimize the risk of file corruption when overwriting
                shutil.move(os.path.join(cfg['experiment_folder'], f"checkpoint_state_new_{patience_metric}.pth.tar"),
                            os.path.join(cfg['experiment_folder'], f"checkpoint_state_{patience_metric}.pth.tar"))
                save_time = time.time() - start_time
                print(f"Completed in {save_time//60:.0f}m {save_time%60:.0f}s.")

                print(f"Best EPOCH for {patience_metric}: {train_cfg[f'epoch_of_best_valid_{patience_metric}']}")
                print(f"Best VALID loss for {patience_metric}: {train_cfg[f'loss_for_best_valid_{patience_metric}']}")
                print(f"Best VALID {patience_metric}: {train_cfg[f'best_valid_{patience_metric}']}")

                mlflow.log_metric(f"best_valid_{patience_metric}", train_cfg[f"best_valid_{patience_metric}"])
                mlflow.log_metric(f"loss_for_best_valid_{patience_metric}", train_cfg[f"loss_for_best_valid_{patience_metric}"])
                if raw_labels == 2:
                    mlflow.log_metric(f"class0_recall_for_best_valid_{patience_metric}", valid_scores["VALID_recall_per_class"][0])
                    mlflow.log_metric(f"class1_recall_for_best_valid_{patience_metric}", valid_scores["VALID_recall_per_class"][1])
                mlflow.log_metric(f"epoch_of_best_valid_{patience_metric}", train_cfg[f"epoch_of_best_valid_{patience_metric}"])

            elif epoch <= 25:
                # Do not stop during the initial plateau caused by the data.
                train_cfg[f"patience_{patience_metric}"] = 0

            else:
                # Number of epochs to accept that validation
                # score hasn't increased or validation loss hasn't decreased
                train_cfg[f"patience_{patience_metric}"] += 1

                if train_cfg[f"patience_{patience_metric}"] >= train_cfg['max_patience']:
                    print(f"Max patience reached for {patience_metric}, stop recording for {patience_metric}")
                    patience_metrics_to_remove.append(patience_metric)

        # Remove patience metrics that reached max patience
        patience_metrics = [i for i in patience_metrics if i not in patience_metrics_to_remove]

        if not patience_metrics:
            print("Max patience reached for all metrics. Training done.")
            break

    # Training done, print statistics
    print(f"\n\nTraining complete in {train_cfg['total_time'] // 60:.0f}m {train_cfg['total_time'] % 60:.0f}s")

    for patience_metric in cfg['training']['patience_metrics']:
        print(f"Best EPOCH for {patience_metric}: {train_cfg[f'epoch_of_best_valid_{patience_metric}']}")
        print(f"Best VALID loss for {patience_metric}: {train_cfg[f'loss_for_best_valid_{patience_metric}']}")
        print(f"Best VALID {patience_metric}: {train_cfg[f'best_valid_{patience_metric}']}\n")

    # Use first patience metric in the list for hyperparameter search criterion
    return train_cfg[f"best_valid_{cfg['training']['patience_metrics'][0]}"]


def batch_loop(loader, model, optimizer, loss_function, device,
               mode="TRAIN", acc_batch_size=None,
               epoch=0, grad_logger=None,
               log_image_sample_every_n_epoch=0):
    """
    Define the mini-batch loop.

    Parameters
    ----------
    loader : obj
        The dataloader to iterate.
    model : obj
        The model to use.
    optimizer : obj
        The optimizer to use.
    loss_function : function
        The loss function to use.
    device : str
        The device that is used. In ['cuda:0', 'cpu']
    mode : str
        The mode to use. In ['TRAIN', 'VALID', 'TEST']
    acc_batch_size : int
        None disables batch accumulation.
        Otherwise, the batch is accumulated to acc_batch_size before doing an update.
        Can only be used in TRAIN mode.
        Default: None
    epoch : int
        Current epoch.
    log_grad_every_n_epoch : int
        Log the histogram of the gradients at every layer individually and
        global stats on the gradients at every N epochs.

    Returns
    -------
    y_true : numpy array
        True labels.
    y_proba : numpy array
        Predicted probabilities.
    epoch_loss : float
        Epoch loss.

    """
    available_modes = ['TRAIN', 'VALID', 'TEST']
    assert mode in available_modes, f"mode can only be {available_modes}"

    # Accumulator for computing metrics
    tot_logits = []
    tot_targets = []

    # If batch accumulation is turned off set the loss and nb_update normally,
    # Otherwise, if it's active and that we are in train mode,
    # Set the proper variable for batch accumulation and override the loss.
    if acc_batch_size is None:
        if mode == "TRAIN":
            # This is needed because in TEST mode optimizer is None
            optimizer.zero_grad()
        nb_updates = len(loader.dataset) // loader.batch_size
        criterion = loss_function

    elif mode == "TRAIN":
        # This is needed because in TEST mode optimizer is None
        optimizer.zero_grad()
        if acc_batch_size % loader.batch_size != 0:
            raise ValueError("batch_size should be a multiple of the acc_batch_size.")

        accumulation_steps = acc_batch_size // loader.batch_size
        # The last incomplete accumulated batch will be dropped if
        # acc_batch_size is not a multiple of the dataset size.
        nb_updates = len(loader.dataset) // acc_batch_size

        # Overriding the loss so it's divided by the accumulation step for batch accumulation.
        def criterion_aggregated_grad(o, t):
            return loss_function(o, t) / accumulation_steps
        criterion = criterion_aggregated_grad
    else:
        raise ValueError("Cannot accumulate gradient when not in TRAIN mode.")

    # Iterate over every batches
    running_loss = 0
    for batch_id, batch in enumerate(loader):
        # Move current batch to device
        inputs = [[view.to(device) for view in eye] for eye in batch['inputs']]
        masks = [mask.to(device) for mask in batch['masks']]
        if loader.dataset.num_class == 1:
            batch['target'] = batch['target'].view(-1, 1)
        targets = batch['target'].to(device)

        # Track history if only in train mode
        with torch.set_grad_enabled(mode == "TRAIN"):
            # Make prediction
            logits = model(inputs, masks)

            # Log results to compute metrics later
            tot_logits += [logits.detach()]
            tot_targets += [targets.detach()]

            # Computing loss
            loss = criterion(logits, targets)
            running_loss += loss.item()

            # Update network if needed
            if mode == "TRAIN":
                # Compute and accumulate gradients.
                loss.backward()

                # Update network and zero-out gradients if aggregated gradients is off
                # Or when it's on, when the appropriate step size is reached.
                if acc_batch_size is None or (batch_id + 1) % accumulation_steps == 0:

                    # Determine if first batch of epoch
                    if acc_batch_size is None:
                        first_batch_of_epoch = batch_id == 0
                    else:
                        first_batch_of_epoch = (batch_id + 1) * loader.batch_size == acc_batch_size

                    if log_image_sample_every_n_epoch > 0:
                        log_sample_epoch = ((epoch - 1) % log_image_sample_every_n_epoch) == 0

                        # Log first view of the primary eye for the whole first batch
                        if first_batch_of_epoch and log_sample_epoch:
                            log_batch_images(batch['inputs'][0][0], epoch)

                    # Log gradient of the first batch of the epoch
                    # If the epoch should be logged
                    if grad_logger.should_log():
                        if first_batch_of_epoch:
                            grad_logger.log_full()
                        else:
                            grad_logger.log()

                    optimizer.step()  # Update the weights
                    optimizer.zero_grad()  # Set all gradients to 0

                    # If batch accumulation is active, drop last batch, IF incomplete.
                    # Stop and do not compute that last mini-batches that would not be updated.
                    if acc_batch_size is not None and \
                       ((batch_id + 1) * loader.batch_size) + acc_batch_size > len(loader.dataset):
                        break

    # Computing epoch metrics
    tot_logits = torch.cat(tot_logits)
    tot_targets = torch.cat(tot_targets)
    if torch.cuda.device_count() > 1 and device != 'cpu':
        outputs_proba = model.module.final_activation(tot_logits)
    else:
        outputs_proba = model.final_activation(tot_logits)

    # Stats to cpu
    epoch_loss = running_loss / nb_updates
    y_true = tot_targets.cpu().numpy()
    y_proba = outputs_proba.cpu().numpy()

    return y_true, y_proba, epoch_loss
