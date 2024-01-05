
from absl import app, flags
from ml_collections import ConfigDict, config_flags

from susie.model import (
    EmaTrainState,
    create_model_def,
    load_pretrained_unet,
    load_text_encoder,
    load_vae,
)


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the hyperparameter configuration.",
    lock_config=False,
)


def main(_):
# def train():
    config = FLAGS.config

    # load vae
    if config.vae is not None:
        vae_encode, vae_decode = load_vae(config.vae)

    # load text encoder
    tokenize, untokenize, text_encode = load_text_encoder(config.text_encoder)


    # load pretrained model
    if config.model.get("pretrained", None):
        pretrained_model_def, pretrained_params = load_pretrained_unet(
            config.model.pretrained, in_channels=12 if config.goal_drop_rate < 1 else 8
        )


if __name__ == "__main__":
    app.run(main)

"""
python3 -u download_models.py --config configs/base.py:base
"""