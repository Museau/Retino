import os
import time
import argparse

import hashlib
import numpy as np
import pickle
import torch

from utils.loss import get_loss
from model.factory import build_model
from utils.config import load_config, load_and_update_eval_config
from utils.checkpoint import load_state_dict, set_seed_and_random_states
from utils.logging import print_and_log_scores
from utils.scores import compute_scores, find_best_threshold

from data.data_loader import get_dataloader
from train import batch_loop


def parse_args():
    """
    Parser for the arguments.

    Returns
    -------
    args : Namespace
        The arguments.

    """
    parser = argparse.ArgumentParser(description="Test one or an ensemble of CNN's")

    parser.add_argument('--eval-config',
                        type=str,
                        help="Path to eval config file.")

    parser.add_argument('--data-folder',
                        type=str,
                        help="Absolute path to the data folder.")

    parser.add_argument('--experiment-folders',
                        nargs='+',
                        default="results/",
                        help="A space-separated list of absolute path to the experiment folder.")

    parser.add_argument('--ignore-cache',
                        dest='ignore_cache',
                        action='store_true',
                        default=False,
                        help="Whether to compute the labels/outputs ignoring evaluation cache.")

    parser.add_argument('-t', '--thresholding',
                        dest='thresholding',
                        action='store_true',
                        default=False,
                        help="Whether to compute the scores using best threshold on validation set.")

    parser.add_argument('-f', '--use-full-dataset',
                        dest='use_full_dataset',
                        action='store_true',
                        default=False,
                        help="Compute scores on the entire given dataset instead of only the test set.")

    args = parser.parse_args()
    return args


def compute_and_print_scores(labels, prediction, splits, scores_for_thresholding, suffix=''):
    """
    Compute and print scores on valid and test.

    Parameters
    ----------
    labels: dict
        test_y_true : array
            Test true labels.
        test_y_true_5_classes : array
            Test true 5 classes labels.
        valid_y_true : array
            Valid true labels.
        valid_y_true_5_classes : array
            Valid true 5 classes labels.
        raw_labels : list
            List of labels.
    prediction: dict
        test_y_proba : array
            Test predicted probabilities.
        valid_y_proba : array
            Valid predicted probabilities.
    splits : list
        List of splits.
    scores_for_thresholding : list
        List of score to use for thresholding.
    suffix : str
        Print suffix (e.g. ' ENSEMBLE'). Default ''.
    """
    threshold = None
    for score_for_thresholding in scores_for_thresholding:
        if score_for_thresholding is not None:
            print(f"\n\nFinding best threshold on VALID{suffix} optimizing for {score_for_thresholding} ...", end=" ")
            start_time = time.time()
            threshold = find_best_threshold(labels['valid_y_true'],
                                            prediction['valid_y_proba'],
                                            raw_labels=labels['raw_labels'],
                                            score_type=score_for_thresholding)
            time_elapsed = time.time() - start_time
            print(f"Completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
            print(f"Best threshold for {score_for_thresholding} on VALID {suffix}: {threshold}")

        # Compute and print score
        for split in splits:
            map_to_binary = False
            print_suffix = suffix
            for i in range(2):  # Needed to map to binary
                map_to_5_classes = len(labels['raw_labels']) == 2
                scores = compute_scores(labels[f'{split}_y_true'],
                                        prediction[f'{split}_y_proba'],
                                        mode=split,
                                        raw_labels=labels['raw_labels'],
                                        threshold=threshold,
                                        map_to_binary=map_to_binary,
                                        map_to_5_classes=map_to_5_classes,
                                        y_true_5_classes=labels[f'{split}_y_true_5_classes'])
                print(f"\nModel {split.upper()}{print_suffix} (thresholding: {str(score_for_thresholding)}) Scores:")
                print_and_log_scores(scores, log=False)
                map_to_binary = len(labels['raw_labels']) > 2
                print_suffix = ' BINARY' + suffix
                if not map_to_binary:
                    break


def _load_and_update_config(args, experiment_folder, verbose):
    # add config to args to be able to load config from given location
    args.config = os.path.join(experiment_folder, 'cfg.yml')
    cfg = load_config(args, test_mode=True)
    cfg['experiment_folder'] = experiment_folder

    # Fix for config pre commit c63337cbc53cc433cea18fb6a24820794ca1e2b9
    if 'feature_scaling' not in cfg['dataset']:
        cfg['dataset']['feature_scaling'] = "MaskedStandardization"

    # Fix for config pre commit 9a4a52d31aeb5d1028c854ea0e92c3c2f3a87ba3
    if 'filter_target' not in cfg['dataset']['train']:
        cfg['dataset']['train']['filter_target'] = False

    # Fix for config pre commit d6b4e98fb3ae34758eca42a3680765b3c99882bc
    if 'patience_metrics' not in cfg['training']:
        cfg['training']['patience_metrics'] = ['quadratic_weighted_kappa']

    # Fix for config pre commit e744fc6e696c4bc1ec7b70421d6445666ff431a7
    if 'n_views' not in cfg['dataset']['train']:
        cfg['dataset']['train']['n_views'] = cfg['dataset']['n_views']
        cfg['dataset']['eval']['n_views'] = cfg['dataset']['n_views']
        cfg['dataset']['eval']['apply_train_augmentations'] = False

    if args.eval_config:
        cfg = load_and_update_eval_config(args.eval_config, cfg, verbose=verbose)

    return cfg


def main_test():
    # Load config file
    args = parse_args()

    # Current device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check that all models in the ensemble have been trained using the same target
    target_names = []
    for experiment_folder in args.experiment_folders:
        cfg = _load_and_update_config(args, experiment_folder, verbose=False)
        target_names.append(cfg['dataset']['target_name'])

    if len(set(target_names)) != 1:
        raise ValueError("target_name should be the same for all model in the ensemble")

    # Train exact is train data set using valid data augmentations
    splits = ['train_exact', 'valid', 'test']

    scores_for_thresholding = [None]  # To compute the score without thresholding
    if args.thresholding:
        scores_for_thresholding += ['kappa', 'f1_macro']
        if cfg['dataset']['target_name'] == 'screening_level':
            scores_for_thresholding += ['J_statistic', 'f1_c1']

    for patience_metric in cfg['training']['patience_metrics']:
        print(f"\n\n#### Model trained using patience metric: {patience_metric} ####")

        prediction = {}
        if len(args.experiment_folders) > 1:
            for split in splits:
                prediction[f'{split}_y_proba_ensemble'] = []

        labels = {}
        for model_num, experiment_folder in enumerate(args.experiment_folders):

            print(f"\n\n#### Model {model_num + 1}/{len(args.experiment_folders)} ####")

            cfg = _load_and_update_config(args, experiment_folder, verbose=True)
            cfg_hash = hashlib.md5(str(cfg).encode()).hexdigest()

            cache_folder = os.path.join(experiment_folder, 'evaluation_cache')
            labels_path = os.path.join(cache_folder, f"{cfg_hash}_labels.pkl")
            prediction_path = os.path.join(cache_folder, f"{cfg_hash}_prediction.pkl")
            labels_prediction_exists = all([os.path.exists(path) for path in [labels_path, prediction_path]])
            if labels_prediction_exists and not args.ignore_cache:
                # Do not recompute labels/prediction
                print(f"\nLoading labels/prediction from the evaluation_cache folder from:\n{cfg['experiment_folder']}")
                # Records labels only once (independent of preprocessing)
                if model_num == 0:
                    with open(labels_path, 'rb') as f:
                        labels = pickle.load(f)
                with open(prediction_path, 'rb') as f:
                    prediction.update(pickle.load(f))

            else:
                # Set seed : needed when we apply transformation at valid/test time.
                set_seed_and_random_states(cfg['seed'], random_states=None, cuda=cfg['cuda'])

                # Set up datasets for each model used in the ensemble as the pre-processing can be different
                loaders = {}
                if args.use_full_dataset:
                    # To save the full dataset prediction (only used when args.use_full_dataset is True)
                    labels['full_dataset_y_true'] = np.array([])
                    labels['full_dataset_y_true_5_classes'] = []
                for split in splits:
                    print(f"\nSetup {cfg['dataset']['name']} resolution {cfg['dataset']['resolution']} {split} data set")
                    loaders[split] = get_dataloader(args.data_folder, cfg['dataset'], split in ['train_exact', 'valid'])[split]
                    print(f"Using feature scaling: {cfg['dataset']['feature_scaling']}")
                    print(f"# of examples: {len(loaders[split].dataset)}")
                    # Records labels only once (independent of preprocessing)
                    if model_num == 0:
                        labels[f'{split}_y_true'] = np.array(loaders[split].dataset.patient_target)
                        labels[f'{split}_y_true_5_classes'] = loaders[split].dataset.patient_target_5_classes
                        if args.use_full_dataset:
                            labels['full_dataset_y_true'] = np.concatenate((labels['full_dataset_y_true'], labels[f'{split}_y_true']))
                            labels['full_dataset_y_true_5_classes'] += labels[f'{split}_y_true_5_classes']
                        # Raw labels are the same regardless of the split
                        if split == 'test':
                            labels['raw_labels'] = loaders['test'].dataset.raw_labels

                # Load model
                state_fname = os.path.join(cfg['experiment_folder'], f"checkpoint_state_{patience_metric}.pth.tar")
                if not os.path.exists(state_fname):
                    # Fix for config pre commit d23d5080bcf04d0989f856955349d3754e3e748e
                    state_fname = os.path.join(cfg['experiment_folder'], 'checkpoint_state.pth.tar')

                if os.path.exists(state_fname):
                    print(f"\nLoading model from:\n{state_fname}")
                    state = load_state_dict(state_fname, True, device)
                    model = build_model(cfg['model'],
                                        loaders['test'].dataset.num_class,
                                        cfg['dataset']['use_both_eyes'],
                                        device,
                                        state['model'])
                    model = model.eval()
                else:
                    raise Exception(f"There is no model weights available in this folder: {cfg['experiment_folder']}")

                # Define loss function
                loss_function = get_loss(cfg['loss']).to(device)

                if args.use_full_dataset:
                    prediction['full_dataset_y_proba'] = np.empty((0, 1))
                for split in splits:
                    print(f"\nEvaluating on {split} set...", end=" ")
                    start_time = time.time()
                    _, prediction[f'{split}_y_proba'], _ = batch_loop(loaders[split],
                                                                      model,
                                                                      None,  # optimizer
                                                                      loss_function,
                                                                      device,
                                                                      mode='TEST')
                    time_elapsed = time.time() - start_time
                    print(f"Completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
                    if args.use_full_dataset:
                        prediction['full_dataset_y_proba'] = np.concatenate((prediction['full_dataset_y_proba'], prediction[f'{split}_y_proba']))

                # Save/overwrite labels/outputs to evaluation_cache
                os.makedirs(cache_folder, exist_ok=True)
                with open(labels_path, 'wb') as f:
                    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(prediction_path, 'wb') as f:
                    pickle.dump(prediction, f, protocol=pickle.HIGHEST_PROTOCOL)

            if args.use_full_dataset:
                # To compute scores on full data set
                splits.append('full_dataset')
            if len(args.experiment_folders) > 1:
                for split in splits:
                    prediction[f'{split}_y_proba_ensemble'].append(prediction[f'{split}_y_proba'])

            compute_and_print_scores(labels, prediction, splits, scores_for_thresholding)

        if len(args.experiment_folders) > 1:

            print(f"\n\n### Evaluating ensemble of {len(args.experiment_folders)} models ###")

            for split in splits:
                # Average the prediction of the different models and update y_proba
                prediction[f'{split}_y_proba'] = np.array(prediction[f'{split}_y_proba_ensemble']).mean(axis=0)

            compute_and_print_scores(labels, prediction, splits, scores_for_thresholding, suffix=' ENSEMBLE')

        splits.remove('full_dataset')


if __name__ == "__main__":
    main_test()
