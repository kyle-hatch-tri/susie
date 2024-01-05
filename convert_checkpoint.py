# import wandb 
import os
import orbax 
from ml_collections import ConfigDict, config_flags
from absl import app, flags

from susie.model import create_model_def

# # load config from wandb
# api = wandb.Api()
# run = api.run(wandb_run_name)
# config = ml_collections.ConfigDict(run.config)


flags.DEFINE_string('checkpoint_path', None, 'Path to checkpoint directory')

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    config = FLAGS.config

    # load params
    params = orbax.checkpoint.PyTreeCheckpointer().restore(os.path.join(FLAGS.checkpoint_path, "params_ema"), item=None)

    # load model
    model_def = create_model_def(config.model)

    # save model
    model_def.save_pretrained(FLAGS.checkpoint_path, params)

if __name__ == "__main__":
    app.run(main)

"""

python3 -u convert_checkpoint.py \
--config configs/base.py:debug \
--checkpoint_path "/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/diffusion_model/jax_model/test1_remote128_2023.12.25_01.41.12/15000"

"""