from __future__ import annotations
import itertools
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import requests
import yaml

from mcli.models.mcli_integration import IntegrationType
from mcli.models.run_input import PartialRunInput, RunInput
from mcli.objects.integrations.git_repo import MCLIGitRepoIntegration
from mcli.serverside.job.mcli_job import MCLIJob
from mcli.serverside.platforms.instance_type import GPUType
from mcli.serverside.runners.runner import Runner

COMPOSER_COMMAND: str = 'composer composer/examples/run_composer_trainer.py -f /mnt/config/parameters.yaml'

# Configure repo
repo: str = 'mosaicml/composer'
branch: str = 'dev'
#repo: str = 'Landanjs/composer'
#branch = 'landan/reflect_padding'
#branch = 'landan/auto_loss_weight'

# Configure system stuff
image: str = 'mosaicml/pytorch_vision:1.11.0_cu113-python3.9-ubuntu20.04'
platform_name: str = 'r1z1'
if platform_name == 'r1z1':
    gpu_type: GPUType = GPUType.A100_80GB  #
elif platform_name == 'r6z2':
    gpu_type: GPUType = GPUType.A100_40GB
gpu_num: int = 8  # Run on 8 GPUs

# Experiment specification
#yaml_path: str = 'dlv3p_ade20k_sprint1/methods/accuratest.yaml'
algorithm = 'sd'
yaml_path = f'dlv3p_ade20k_sprint1/methods/sample_{algorithm}.yaml'
wandb_project_name = f"dlv3p-ade20k-{algorithm}"
#wandb_group_name: str = ''

# Add git repo
integrations: List[Dict[str, Any]] = []
composer_repo = MCLIGitRepoIntegration(
    integration_type=IntegrationType.git_repo,
    git_repo=repo,
    git_branch=branch,
    pip_install='--user -e .[all]',
)
integrations.append(asdict(composer_repo))

# Hyperparameters to sweep

with open(yaml_path, 'r') as f:
    parameters = yaml.safe_load(f)
#scale_schedule_ratios = [0.25, 0.33, 0.5, 0.75, 1.0]
scale_schedule_ratios = [1.0]
seeds = [21, 42, 17, 19]
hparams = itertools.product(scale_schedule_ratios, seeds,
                            ['uniform', 'linear'], [0.05, 0.1, 0.2])
for hparam in hparams:
    ssr, seed, hparam1, hparam2 = hparam
    print(f"New params:{hparam}")
    parameters["scale_schedule_ratio"] = ssr
    parameters["seed"] = seed
    parameters['algorithms']['stochastic_depth']['drop_distribution'] = hparam1
    parameters['algorithms']['stochastic_depth']['drop_rate'] = hparam2
    parameters["loggers"]["wandb"]["project"] = wandb_project_name
    parameters["loggers"]["wandb"][
        "group"] = f'{algorithm}-{hparam1}-{hparam2}'
    print(parameters)

    base_run = PartialRunInput(
        run_name='deeplabv3-baseline',
        platform=platform_name,
        gpu_num=gpu_num,
        gpu_type=gpu_type.value,
        image=image,
        integrations=integrations,
        command=COMPOSER_COMMAND,
        parameters=parameters,
    )

    runner = Runner()
    run_input = RunInput.from_partial_run_input(base_run)
    job = MCLIJob.from_run_input(run_input)
    if platform_name == 'r1z1':
        runner.submit(job, priority_class='scavenge')
    else:
        runner.submit(job)
