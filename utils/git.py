import os

import git
import mlflow


def get_repo_state():
    """
    Returns state of the git repository which contains:
    - The active branch name
    - The hash of the active commit
    - The hash of the last pushed commit
    - If there are some uncommited changes

    Returns
    -------
        state: dict
            Dictionary of Git status info.
    """
    repo = git.Repo(search_parent_directories=True)
    branch = repo.active_branch

    state = {"Git - Active branch name": branch.name}
    state["Git - Active commit"] = branch.commit.hexsha
    state["Git - Uncommited Changes"] = repo.is_dirty()

    remote_name = f"origin/{branch.name}"
    if remote_name in repo.refs:
        state["Git - Last pushed commit"] = repo.refs[remote_name].commit.hexsha
    else:
        state["Git - Last pushed commit"] = "UNPUSHED BRANCH"
    return state


def log_git_diff(experiment_folder, mlflow_dir):
    """
    Log the git diff to mlflow active run as an artifact
    and to the current experiment_folder.

    Parameters
    ----------
    experiment_folder: str
        Path to the current experiment folder.
    """
    repo = git.Repo(search_parent_directories=True)
    diff = repo.git.diff()

    diff_path = os.path.join(experiment_folder, "git_diff.txt")
    with open(diff_path, "w") as f:
        f.write(diff)
    mlflow.log_artifact(diff_path, mlflow_dir)
