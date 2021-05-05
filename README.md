# Diabetic retinopathy

## Environment
1. Download miniconda: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
2. Install miniconda: `bash Miniconda3-latest-Linux-x86_64.sh`
3. To create the environment to run the model, simply do `conda env create -f <path to>/environment.yml`.

You can find the appropriate `environment.yml` [here](environment.yml).

### Cluster Environment
To have an efficient environment on SLURM clusters, you can _pack_ your environment and _unpack_ it at the beginning of you job on the local SSD.

Assuming you installed the environment in the section above and then activated it.
The following command will generate `retino.tar.gz` which will contain your entire environment.

`conda pack -n retino`

Then add the following snippet at the beginning of your SLURM launch script and you should be all set.
```bash
# Make sure no previous conda env is activated
if command -v conda; then
  conda deactivate
fi

# Create env folder
ENV_PATH=$SLURM_TMPDIR/retino
mkdir -p $ENV_PATH

# Extract packed env
tar -xzf <path to>/retino.tar.gz -C $ENV_PATH

# Activate the environment. This adds `$ENV_PATH/bin` to your path
source $ENV_PATH/bin/activate

# Cleanup prefixes from in the active environment.
conda-unpack
```

For more information see [the conda-pack documentation](https://conda.github.io/conda-pack/#commandline-usage).


### Git Hooks Activation
Make sure you have git version > 2.9.
After cloning, run the following command once to activate the proper hooks.

From the root of the repo run `git config core.hooksPath .githooks`.

## Config file
The hyperparameters to use are defined in a yaml config file, see [config.yml](config.yml) for an example, and are passed to the model using the `--config` flag.

Note that in the `model/base` sub-config the normalization layer `norm_layer` hyperparameter is fully dynamical and assumes you know which parameter to use with which norm_layer/model combination. __No checks are made__. For example, if you use `group_norm` with `resnet`, you need to also pass `num_groups` and it needs to be a divider of 64.


## Hyperparameter search (Orion)
### Setup
Install orion in your env.
`pip install orion`

### Config
You need to specify the [search spaces](https://orion.readthedocs.io/en/stable/user/searchspace.html) in the config file. See [orion.example.conf.yml](orion.example.conf.yml) for an example.

The [templating](https://orion.readthedocs.io/en/stable/user/script.html#command-line-templating) used in the config file is [manually and only partially](utils/config.py#L52) supported.

### Run
Then to launch a job in the specified search space using the following command.

`orion hunt -c .orionconfig.yml -n kaggle --worker-max-trials 1 python main.py --data-folder data/kaggle2015/ --experiment-folder results-eye --config orion.example.conf.yml`

To run orion on a [slurm cluster](https://orion.readthedocs.io/en/stable/tutorials/cluster.html) see the documentation of the [included example](launch_orion_slurm_example.sh).
