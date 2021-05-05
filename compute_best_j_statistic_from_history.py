import os
import glob
import argparse

import ruamel.yaml as yaml
import mlflow


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Update artifact path of mlruns.")

    parser.add_argument('mlruns_path',
                        type=str,
                        help="Path to mlruns folder to update.")

    args = parser.parse_args()
    return args


def load_metric_history(path_to_metric_history):
    with open(path_to_metric_history) as f:
        metric_history = [[eval(elem) for elem in i.strip().split(' ')] for i in f.readlines()]

    return metric_history


def get_metric_for_epoch(path_to_metric_history, epoch):
    metric_history = load_metric_history(path_to_metric_history)
    metric_for_epoch = [i[1] for i in metric_history if i[2] == epoch][0]
    return metric_for_epoch


def get_best_metric(run_id, valid_metric_history):
    """
    Get best metric using early stopping policy in validation metric history.

    Parameters
    ----------
    run_id : str
        Id of the run to be updated.
    valid_metric_history : list of tuple
        List of tuple containing the valid metric history under the form
        [(valid metric, epoch), ...].

    Returns
    -------
    best_valid_metric : float
        Best valid metric.
    best_epoch_metric : int
        Best epoch metric.
    """
    run = mlflow.get_run(run_id=run_id)

    # Apply early stopping policy to get best metric
    patience = 0
    max_patience = int(run.data.params['training.max_patience'])  # Get max patience from mlflow run data
    best_valid_metric = 0.
    best_epoch_metric = 1  # Start at epoch 1 in case best_metric is always 0.
    best_epoch_time = valid_metric_history[0][0]  # Start at epoch 1 time
    for time, valid_metric, epoch in valid_metric_history:
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            best_epoch_metric = epoch
            best_epoch_time = time
            patience = 0

        elif epoch <= 25:
            patience = 0

        else:
            patience += 1
            if patience == max_patience:
                break

    return best_valid_metric, best_epoch_metric, best_epoch_time


if __name__ == '__main__':
    args = parse_args()

    # Set mlflow tracking uri
    mlruns_path = os.path.realpath(args.mlruns_path)
    mlflow.set_tracking_uri(mlruns_path)

    # Get experiment numbers and run ids
    exp_and_run_ids = [(run_id.split('/')[-3], run_id.split('/')[-2]) for run_id in glob.glob(f'{mlruns_path}/*/*/')]

    exp_run_to_rerun = []
    exp_run_with_no_history = []
    for exp_id, run_id in exp_and_run_ids:

        # Record if j-statistic history exist in mlflow folder for the run
        path_to_run = f'{mlruns_path}/{exp_id}/{run_id}'
        path_to_valid_j_statistic_history = f'{path_to_run}/metrics/VALID_j_statistic'
        path_to_valid_j_statistic_history_exist = os.path.isfile(path_to_valid_j_statistic_history)
        # Check if patience metric did not include j_statistic
        no_patience_j_statistic = not(os.path.isfile(f'{path_to_run}/metrics/best_valid_j_statistic'))

        # Record if active run
        meta_data_path = f'{path_to_run}/meta.yaml'
        with open(meta_data_path, 'r') as f:
            meta_data = yaml.load(f, Loader=yaml.Loader)
        active_lifecyle = (meta_data['lifecycle_stage'] == "active")

        if path_to_valid_j_statistic_history_exist and active_lifecyle and no_patience_j_statistic:
            # Get best j-statistic and best j-statistic epoch
            valid_j_statistic_history = load_metric_history(path_to_valid_j_statistic_history)
            best_valid_j_statistic, epoch_of_best_valid_j_statistic, epoch_of_best_valid_j_statistic_time = get_best_metric(run_id, valid_j_statistic_history)

            # Find associated class0/class1 recall (i.e. specificity/sensitivity) for best j-statistic
            class0_recall_for_best_valid_j_statistic = get_metric_for_epoch(f'{path_to_run}/metrics/VALID_class0_recall', epoch_of_best_valid_j_statistic)
            class1_recall_for_best_valid_j_statistic = get_metric_for_epoch(f'{path_to_run}/metrics/VALID_class1_recall', epoch_of_best_valid_j_statistic)
            # Find associated loss
            loss_for_best_valid_j_statistic = get_metric_for_epoch(f'{path_to_run}/metrics/VALID_loss', epoch_of_best_valid_j_statistic)

            # Record experiment to rerun and add a j_statistic_early_stopping_incomplete flag to mlflow in metrics
            last_epoch_recorded = valid_j_statistic_history[-1][-1]
            j_statistic_early_stopping_incomplete = 0
            if last_epoch_recorded == epoch_of_best_valid_j_statistic:
                exp_run_to_rerun.append((exp_id, run_id))
                j_statistic_early_stopping_incomplete = 1

            # Update mlflow for best valid j_statistic
            file_names = ['best_valid_j_statistic',
                          'class0_recall_for_best_valid_j_statistic',
                          'class1_recall_for_best_valid_j_statistic',
                          'loss_for_best_valid_j_statistic',
                          'epoch_of_best_valid_j_statistic',
                          'j_statistic_early_stopping_incomplete']
            for file_name in file_names:
                file = open(f"{path_to_run}/metrics/{file_name}", "a")
                file.write(f"{epoch_of_best_valid_j_statistic_time} {eval(file_name)} 0")
                file.close()

            # Update mlflow for best quadratic weighted kappa
            with open(f"{path_to_run}/metrics/best_epoch") as f:
                best_epochs = f.readlines()
            best_epoch_valid_quadratic_weighted_kappa = int(best_epochs[-1].split(' ')[1])
            best_epoch_valid_quadratic_weighted_kappa_time = int(best_epochs[-1].split(' ')[0])

            # Find associated class0/class1 recall (i.e. specificity/sensitivity) for best quadratic weighted kappa
            class0_recall_for_best_valid_quadratic_weighted_kappa = get_metric_for_epoch(f'{path_to_run}/metrics/VALID_class0_recall', best_epoch_valid_quadratic_weighted_kappa)
            class1_recall_for_best_valid_quadratic_weighted_kappa = get_metric_for_epoch(f'{path_to_run}/metrics/VALID_class1_recall', best_epoch_valid_quadratic_weighted_kappa)
            file_names = ['class0_recall_for_best_valid_quadratic_weighted_kappa', 'class1_recall_for_best_valid_quadratic_weighted_kappa']
            for file_name in file_names:
                file = open(f"{path_to_run}/metrics/{file_name}", "a")
                file.write(f"{best_epoch_valid_quadratic_weighted_kappa_time} {eval(file_name)} 0")

            os.rename(f"{path_to_run}/metrics/best_epoch", f"{path_to_run}/metrics/epoch_of_best_valid_quadratic_weighted_kappa")
            os.rename(f"{path_to_run}/metrics/best_valid_loss", f"{path_to_run}/metrics/loss_for_best_valid_quadratic_weighted_kappa")

    print(f'# of runs to rerun/# of total run: {len(exp_run_to_rerun)}/{len(exp_and_run_ids)}\n')
    print(f'Experiment/run to rerun: {exp_run_to_rerun}')
