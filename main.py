import os
import sys
import time
import argparse

import torch
import mlflow

from orion.client.cli import report_objective

from train import train
from model.factory import get_model
from data.data_loader import get_dataloader
from utils.loss import get_loss
from utils.model import print_summary
from utils.optimizer import get_optimizer
from utils.logging import log_dict_to_artefact, log_config
from utils.lr_scheduler import get_lr_scheduler
from utils.git import get_repo_state, log_git_diff
from utils.config import update_config_and_setup_exp_folder, load_config
from utils.checkpoint import load_checkpoint, set_seed_and_random_states, update_state


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    """
    Parser for the command line arguments.

    Returns
    -------
    args : Namespace
        The arguments.
    """
    parser = argparse.ArgumentParser(description="Train a CNN network")
    parser.add_argument('--config',
                        type=str,
                        help="Config file.")

    parser.add_argument('--data-folder',
                        type=str,
                        required=True,
                        help="Absolute path to the data folder.")

    parser.add_argument('--experiment-folder',
                        type=str,
                        default="results/",
                        help="Absolute path to the result folder.")

    parser.add_argument('-f', '--force-resume',
                        dest='force_resume',
                        action='store_true',
                        default=False,
                        help="Skip proper verification and force resuming from given config.")

    parser.add_argument('-n', '--no-resume',
                        dest='no_resume',
                        action='store_true',
                        default=False,
                        help="Ignore resuming and overwrites/restarts the experiment from the beginning.")

    args = parser.parse_args()

    if args.force_resume and args.no_resume:
        parser.error('Cannot use --force-resume and --no-resume at the same time.')

    return args


def main():
    args = parse_args()
    cfg = load_config(args)

    # Setting device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    mlflow.set_experiment(experiment_name=cfg['experiment_name'])
    # If run_id is None MLFlow start a new run otherwise it resumes.
    with mlflow.start_run(run_id=cfg.get('mlflow_uid')) as ml_run:
        cfg = update_config_and_setup_exp_folder(args, cfg, mlflow.active_run().info.run_id)

        # Log to MLFlow if in a special mode.
        if args.force_resume:
            mlflow.log_param("Mode", "Forced Resume")
        elif args.no_resume:
            mlflow.log_param("Mode", "No Resume")

        # Load checkpoint state
        state = load_checkpoint(cfg['experiment_folder'], device)
        if args.force_resume:
            state = update_state(state, cfg)

        # Log git repo state to MLFlow
        repo_state = get_repo_state()

        # Get previous git state and make sure it's the same before resuming.
        # This enforce reproducibility
        active_run_params = ml_run.data.params
        if not args.force_resume:
            for g in active_run_params:
                if g.startswith("Git -"):
                    if str(repo_state[g]) != str(active_run_params[g]):
                        err_message = "Cannot resume experiment from different code.\n"
                        err_message += f"{g} does not match. {repo_state[g]} => {active_run_params[g]}"
                        raise Exception(err_message)
                    elif g == "Git - Uncommited Changes" and repo_state[g] is True:
                        err_message = "Cannot resume experiment that was started with uncommited code."
                        raise Exception(err_message)

        if args.force_resume and active_run_params != {}:
            mlflow_dir = f"force_resumed/{time.strftime('%Y_%m_%d_%H%M%S')}"
            log_dict_to_artefact(repo_state, "repo_state.yml", mlflow_dir)
            log_dict_to_artefact(cfg, "cfg.yml", mlflow_dir)
        else:
            mlflow_dir = None
            mlflow.log_params(repo_state)
            log_config(cfg)

        if repo_state["Git - Uncommited Changes"]:
            print("Warning : Some uncommited changes were detected. The only way to resume this experiment will be to use --force-resume.")
            log_git_diff(cfg['experiment_folder'], mlflow_dir)

        # Set seed
        set_seed_and_random_states(cfg['seed'], None, cfg['cuda'])

        # Load data
        data_cfg = cfg['dataset']
        print(f"Setup {data_cfg['name']} resolution {data_cfg['resolution']} train/valid data set")
        data_loaders = get_dataloader(args.data_folder, data_cfg, True)
        print(f"# of train examples: {len(data_loaders['train'].dataset)}")
        print(f"# of valid examples: {len(data_loaders['valid'].dataset)}")

        # Define/Load model
        model = get_model(cfg['model'],
                          data_loaders['train'].dataset.num_class,
                          data_cfg['use_both_eyes'],
                          device,
                          state['model'],
                          cfg['training']['grad_checkpointing'])
        print_summary(model)

        # Define optimizer
        optimizer = get_optimizer(cfg['optimizer'], model, state['optim'])
        print(f"\nUsing Optimizer:\n{optimizer}")

        # Define lr scheduler
        lr_scheduler = get_lr_scheduler(cfg['lr_scheduler'], optimizer, state['scheduler'])
        if lr_scheduler:
            print(f"Using lr scheduler:\n\t{lr_scheduler.__class__.__name__}\n\t{lr_scheduler.__dict__}")

        # Define loss function
        loss_function = get_loss(cfg['loss']).to(device)
        print(f"\nUsing Loss:\n{loss_function}")

        # Training
        score = train(data_loaders, model, optimizer, lr_scheduler, loss_function, state, cfg, device)
        report_objective(-score)


if __name__ == '__main__':
    main()
