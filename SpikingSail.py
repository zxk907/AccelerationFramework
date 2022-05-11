'''
Copyright 2020 The Microsoft DeepSpeed Team
'''

import deepspeed

import sys
import types

from  deepspeed import ops

from deepspeed import runtime

from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.runtime.engine import ADAM_OPTIMIZER, LAMB_OPTIMIZER
from deepspeed.runtime.pipe.engine import PipelineEngine
from deepspeed.runtime.lr_schedules import add_tuning_arguments
from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError
from deepspeed.runtime.activation_checkpointing import checkpointing
from deepspeed.ops.transformer import DeepSpeedTransformerLayer, DeepSpeedTransformerConfig
from deepspeed.utils import log_dist
from deepspeed.utils.distributed import init_distributed

from deepspeed.runtime import zero

from deepspeed.pipe import PipelineModule

from deepspeed.git_version_info import version, git_hash, git_branch


def _parse_version(version_str):
    '''Parse a version string and extract the major, minor, and patch versions.'''
    import re
    matched = re.search('^(\d+)\.(\d+)\.(\d+)', version_str)
    return int(matched.group(1)), int(matched.group(2)), int(matched.group(3))


# Export version information
__version__ = version
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

# Provide backwards compatability with old deepspeed.pt module structure, should hopefully not be used
pt = types.ModuleType('pt', 'dummy pt module for backwards compatability')
deepspeed = sys.modules[__name__]
setattr(deepspeed, 'pt', pt)
setattr(deepspeed.pt, 'deepspeed_utils', deepspeed.runtime.utils)
sys.modules['deepspeed.pt'] = deepspeed.pt
sys.modules['deepspeed.pt.deepspeed_utils'] = deepspeed.runtime.utils
setattr(deepspeed.pt, 'deepspeed_config', deepspeed.runtime.config)
sys.modules['deepspeed.pt.deepspeed_config'] = deepspeed.runtime.config
setattr(deepspeed.pt, 'loss_scaler', deepspeed.runtime.fp16.loss_scaler)
sys.modules['deepspeed.pt.loss_scaler'] = deepspeed.runtime.fp16.loss_scaler


class useDeepspeed():
    def __init__(self, args, model,  optimizer):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
    
    
    def run(self):
        model_engine, optimizer, _, _ = initialize(args = self.args, model=self.model, model_parameters=self.model.parameters(),optimizer=self.optimizer)
        return model_engine, optimizer

def initialize(args=None,
               model=None,
               optimizer=None,
               model_parameters=None,
               training_data=None,
               lr_scheduler=None,
               mpu=None,
               dist_init_required=None,
               collate_fn=None,
               config=None,
               config_params=None):
    
    log_dist("DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
        __version__,
        __git_hash__,
        __git_branch__),
             ranks=[0])

    assert model is not None, "deepspeed.initialize requires a model"

    if not isinstance(model, PipelineModule):
        engine = DeepSpeedEngine(args=args,
                                 model=model,
                                 optimizer=optimizer,
                                 model_parameters=model_parameters,
                                 training_data=training_data,
                                 lr_scheduler=lr_scheduler,
                                 mpu=mpu,
                                 dist_init_required=dist_init_required,
                                 collate_fn=collate_fn,
                                 config=config,
                                 config_params=config_params)
    else:
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(args=args,
                                model=model,
                                optimizer=optimizer,
                                model_parameters=model_parameters,
                                training_data=training_data,
                                lr_scheduler=lr_scheduler,
                                mpu=model.mpu(),
                                dist_init_required=dist_init_required,
                                collate_fn=collate_fn,
                                config=config,
                                config_params=config_params)

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler
    ]
    return tuple(return_items)


def _add_core_arguments(parser):
   
    group = parser.add_argument_group('DeepSpeed', 'DeepSpeed configurations')

    group.add_argument(
        '--deepspeed',
        default=False,
        action='store_true',
        help=
        'Enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)')

    group.add_argument('--deepspeed_config',
                       default=None,
                       type=str,
                       help='DeepSpeed json configuration file.')

    group.add_argument(
        '--deepscale',
        default=False,
        action='store_true',
        help=
        'Deprecated enable DeepSpeed (helper flag for user code, no impact on DeepSpeed backend)'
    )

    group.add_argument('--deepscale_config',
                       default=None,
                       type=str,
                       help='Deprecated DeepSpeed json configuration file.')

    group.add_argument(
        '--deepspeed_mpi',
        default=False,
        action='store_true',
        help=
        "Run via MPI, this will attempt to discover the necessary variables to initialize torch "
        "distributed from the MPI environment")

    return parser


def add_config_arguments(parser):
    
    parser = _add_core_arguments(parser)

    return parser


