"""Unified GPT XTTS fine-tuning script for LJSpeech (Voice.Me).

This script supports both XTTS v1 and XTTS v2 model training via a single
entry point.  Pass ``--version v1`` or ``--version v2`` (default) to select
the model variant.  Version-specific behaviour includes:

* **v1** -- Uses XTTS-v1 model files, 8194 audio tokens, and the original
  GPT architecture without perceiver resampler or masked ground-truth prompt
  approach.
* **v2** -- Uses XTTS-v2 model files, 1026 audio tokens, and enables both
  ``gpt_use_masking_gt_prompt_approach`` and
  ``gpt_use_perceiver_resampler``.

Example usage::

    # Fine-tune with XTTS v2 (default)
    python train_gpt_xtts.py

    # Fine-tune with XTTS v1
    python train_gpt_xtts.py --version v1
"""

import argparse
import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# ---------------------------------------------------------------------------
# Version-specific configuration
# ---------------------------------------------------------------------------

VERSION_CONFIGS = {
    "v1": {
        "run_name": "GPT_XTTS_LJSpeech_FT",
        "checkpoints_dir": "XTTS_v1.1_original_model_files",
        "dvae_url": "https://models.voiceme.ai/xtts-v1/dvae.pth",
        "mel_norm_url": "https://models.voiceme.ai/xtts-v1/mel_stats.pth",
        "tokenizer_url": "https://models.voiceme.ai/xtts-v1/vocab.json",
        "model_url": "https://models.voiceme.ai/xtts-v1/model.pth",
        "gpt_num_audio_tokens": 8194,
        "gpt_start_audio_token": 8192,
        "gpt_stop_audio_token": 8193,
        "gpt_use_masking_gt_prompt_approach": False,
        "gpt_use_perceiver_resampler": False,
        "download_label": "XTTS v1.1",
    },
    "v2": {
        "run_name": "GPT_XTTS_v2.0_LJSpeech_FT",
        "checkpoints_dir": "XTTS_v2.0_original_model_files",
        "dvae_url": "https://models.voiceme.ai/xtts-v2/dvae.pth",
        "mel_norm_url": "https://models.voiceme.ai/xtts-v2/mel_stats.pth",
        "tokenizer_url": "https://models.voiceme.ai/xtts-v2/vocab.json",
        "model_url": "https://models.voiceme.ai/xtts-v2/model.pth",
        "gpt_num_audio_tokens": 1026,
        "gpt_start_audio_token": 1024,
        "gpt_stop_audio_token": 1025,
        "gpt_use_masking_gt_prompt_approach": True,
        "gpt_use_perceiver_resampler": True,
        "download_label": "XTTS v2.0",
    },
}

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PROJECT_NAME = "XTTS_trainer"
DASHBOARD_LOGGER = "tensorboard"
LOGGER_URI = None

# Set here the path that the checkpoints will be saved. Default: ./run/training/
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run", "training")

# Training Parameters
OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # for multi-gpu training please make it False
START_WITH_EVAL = True  # if True it will start with evaluation
BATCH_SIZE = 3  # set here the batch size
GRAD_ACUMM_STEPS = 84  # set here the grad accumulation steps
# Note: we recommend that BATCH_SIZE * GRAD_ACUMM_STEPS need to be at least
# 252 for more efficient training.  You can increase/decrease BATCH_SIZE but
# then set GRAD_ACUMM_STEPS accordingly.

# Dataset paths - set via environment variables or modify directly
DATASET_PATH = os.environ.get("DATASET_PATH", "/path/to/LJSpeech-1.1_24khz/")

# Define here the dataset that you want to use for the fine-tuning on.
config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="ljspeech",
    path=DATASET_PATH,
    meta_file_train=os.path.join(DATASET_PATH, "metadata.csv"),
    language="en",
)

# Add here the configs of the datasets
DATASETS_CONFIG_LIST = [config_dataset]

# Training sentences generations
SPEAKER_REFERENCE = [
    "./tests/data/ljspeech/wavs/LJ001-0002.wav"  # speaker reference to be used in training test sentences
]
LANGUAGE = config_dataset.language


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_downloaded(urls, dest_dir, label):
    """Download model files into *dest_dir* if they are not already present."""
    missing = [u for u in urls if not os.path.isfile(os.path.join(dest_dir, os.path.basename(u)))]
    if missing:
        print(f" > Downloading {label} files!")
        ModelManager._download_model_files(missing, dest_dir, progress_bar=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT XTTS on LJSpeech (Voice.Me)")
    parser.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v2",
        help="XTTS model version to train (default: v2)",
    )
    args = parser.parse_args()

    vcfg = VERSION_CONFIGS[args.version]

    # Resolve checkpoint output directory
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, vcfg["checkpoints_dir"])
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # URLs
    DVAE_CHECKPOINT_LINK = vcfg["dvae_url"]
    MEL_NORM_LINK = vcfg["mel_norm_url"]
    TOKENIZER_FILE_LINK = vcfg["tokenizer_url"]
    XTTS_CHECKPOINT_LINK = vcfg["model_url"]

    # Local paths (derived from URLs)
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

    # Download files if needed
    _ensure_downloaded(
        [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK],
        CHECKPOINTS_OUT_PATH,
        f"{vcfg['download_label']} DVAE",
    )
    _ensure_downloaded(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK],
        CHECKPOINTS_OUT_PATH,
        vcfg["download_label"],
    )

    # Build GPTArgs -- start with the common keyword arguments, then layer on
    # the version-specific ones.
    gpt_kwargs = dict(
        max_conditioning_length=132300,  # 6 secs
        min_conditioning_length=66150,  # 3 secs
        debug_loading_failures=False,
        max_wav_length=255995,  # ~11.6 seconds
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=vcfg["gpt_num_audio_tokens"],
        gpt_start_audio_token=vcfg["gpt_start_audio_token"],
        gpt_stop_audio_token=vcfg["gpt_stop_audio_token"],
    )

    if vcfg["gpt_use_masking_gt_prompt_approach"]:
        gpt_kwargs["gpt_use_masking_gt_prompt_approach"] = True
    if vcfg["gpt_use_perceiver_resampler"]:
        gpt_kwargs["gpt_use_perceiver_resampler"] = True

    model_args = GPTArgs(**gpt_kwargs)

    # Define audio config
    audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

    # Training parameters config
    config = GPTTrainerConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=vcfg["run_name"],
        project_name=PROJECT_NAME,
        run_description="""
            GPT XTTS training
            """,
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=10000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        # target_loss="loss",
        print_eval=False,
        # Optimizer values like tortoise, pytorch implementation with
        # modifications to not apply WD to non-weight parameters.
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        # it was adjusted accordingly for the new step scheme
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[
            {
                "text": "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
            {
                "text": "This cake is great. It's so delicious and moist.",
                "speaker_wav": SPEAKER_REFERENCE,
                "language": LANGUAGE,
            },
        ],
    )

    # Init the model from config
    model = GPTTrainer.init_from_config(config)

    # Load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Init the trainer and start training
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
