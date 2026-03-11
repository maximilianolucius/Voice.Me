# Voice.Me - Text-to-Speech Toolkit

A deep learning toolkit for Text-to-Speech (TTS), battle-tested in research and production.

## Features

- **14+ TTS Models**: Tacotron, Tacotron2, VITS, GlowTTS, ForwardTTS, AlignTTS, DelightfulTTS, NeuralHMM, OverFlow, Bark, Tortoise, XTTS, and more.
- **8+ Vocoder Models**: HiFiGAN, WaveGrad, WaveRNN, MelGAN, MultiBand MelGAN, ParallelWaveGAN, UnivNet.
- **Speaker Encoders**: LSTM and ResNet-based speaker encoders for multi-speaker TTS and voice cloning.
- **Voice Conversion**: FreeVC for zero-shot voice cloning.
- **Multi-language Support**: English, German, French, Spanish, Chinese, Korean, Japanese, Belarusian, Bangla, and more.
- **Cross-lingual Voice Cloning**: XTTS supports 17+ languages with reference audio.
- **Easy API**: Simple Python API and CLI for synthesis, training, and fine-tuning.
- **Web Server**: Flask-based HTTP server for TTS-as-a-service.
- **PyTorch Hub**: Load models directly via `torch.hub`.

## Quick Start

### Installation

```bash
pip install -e .
```

### Synthesize Speech (Python API)

```python
from TTS.api import TTS

# List available models
print(TTS().list_models())

# Initialize TTS with a model
tts = TTS(model_name="tts_models/en/ljspeech/vits")

# Synthesize to file
tts.tts_to_file(text="Hello world!", file_path="output.wav")

# Synthesize with voice cloning
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.tts_to_file(
    text="Hello world!",
    speaker_wav="reference.wav",
    language="en",
    file_path="output.wav",
)
```

### Synthesize Speech (CLI)

```bash
# List models
voiceme-tts --list_models

# Synthesize with a pre-trained model
voiceme-tts --model_name "tts_models/en/ljspeech/vits" \
            --text "Hello world!" \
            --out_path output.wav

# Start the web server
voiceme-server --model_name "tts_models/en/ljspeech/vits"
```

### Training

```bash
# Train a TTS model
python TTS/bin/train_tts.py --config_path config.json

# Train a vocoder
python TTS/bin/train_vocoder.py --config_path config.json

# Train a speaker encoder
python TTS/bin/train_encoder.py --config_path config.json
```

## Project Structure

```
Voice.Me/
├── TTS/                        # Main package
│   ├── api.py                  # Public API (tts(), tts_to_file())
│   ├── model.py                # Abstract base model class
│   ├── server/                 # Flask HTTP server
│   ├── bin/                    # CLI scripts (train, synthesize, compute, etc.)
│   ├── config/                 # Shared configuration classes
│   ├── encoder/                # Speaker/emotion encoder models
│   │   ├── models/             # LSTM, ResNet encoders
│   │   ├── configs/            # Encoder configurations
│   │   └── utils/              # Training, visualization utilities
│   ├── tts/                    # TTS models (core)
│   │   ├── models/             # All TTS model implementations
│   │   ├── layers/             # Neural network layers per model
│   │   ├── configs/            # Model-specific configurations
│   │   ├── datasets/           # Dataset loading and formatting
│   │   └── utils/              # Text processing, phonemizers, synthesis
│   ├── vc/                     # Voice conversion (FreeVC)
│   ├── vocoder/                # Vocoder models and layers
│   │   ├── models/             # HiFiGAN, WaveGrad, WaveRNN, MelGAN, etc.
│   │   ├── layers/             # Vocoder-specific layers
│   │   ├── configs/            # Vocoder configurations
│   │   └── datasets/           # Audio dataset handling
│   ├── utils/                  # Shared utilities
│   │   └── audio/              # Audio processing (numpy, torch transforms)
│   └── demos/                  # Gradio demo for XTTS fine-tuning
├── tests/                      # Test suites by component
│   ├── tts_tests/              # TTS model tests
│   ├── vocoder_tests/          # Vocoder tests
│   ├── text_tests/             # Text processing tests
│   ├── data_tests/             # Data loading tests
│   ├── aux_tests/              # Audio processor, speaker encoder tests
│   ├── inference_tests/        # Synthesis/inference tests
│   ├── xtts_tests/             # XTTS-specific tests
│   ├── vc_tests/               # Voice conversion tests
│   └── zoo_tests/              # Pre-trained model zoo tests
├── recipes/                    # Training recipes per dataset/model
│   ├── ljspeech/               # LJSpeech recipes
│   ├── vctk/                   # VCTK multi-speaker recipes
│   ├── multilingual/           # Multilingual recipes
│   └── thorsten_DE/            # German Thorsten recipes
├── docs/                       # Sphinx documentation
├── pyproject.toml              # Project configuration and dependencies
└── requirements.txt            # Legacy pip requirements
```

## Supported Models

### Text-to-Speech

| Model | Type | Multi-Speaker | Description |
|-------|------|:---:|-------------|
| Tacotron | Autoregressive | Yes | Original sequence-to-sequence TTS |
| Tacotron2 | Autoregressive | Yes | Improved Tacotron with mel-spectrogram prediction |
| VITS | End-to-End | Yes | Flow-based with stochastic duration predictor |
| GlowTTS | Flow-based | Yes | Parallel synthesis with learned alignments |
| ForwardTTS | Parallel | Yes | Duration-based with optional pitch/energy |
| FastPitch | Parallel | Yes | Pitch-conditioned parallel TTS |
| FastSpeech2 | Parallel | Yes | Variance adaptor for pitch, energy, duration |
| AlignTTS | Parallel | No | Transformer encoder with MDN alignment |
| SpeedySpeech | Parallel | No | Lightweight parallel TTS |
| DelightfulTTS | Parallel | Yes | Conformer-based with F0 and HiFiGAN |
| NeuralHMM | HMM-based | No | Neural networks + hidden Markov models |
| OverFlow | HMM + Flow | No | Neural HMM with normalizing flows |
| Bark | Token-based | Yes | GPT + EnCodec discrete token synthesis |
| Tortoise | Multi-stage | Yes | Autoregressive + diffusion + CLVP |
| XTTS | GPT-based | Yes | Cross-lingual voice cloning (17+ languages) |

### Vocoders

| Model | Type | Description |
|-------|------|-------------|
| HiFiGAN | GAN | High-fidelity waveform generation |
| WaveGrad | Diffusion | Score-matching diffusion vocoder |
| WaveRNN | Autoregressive | Recurrent waveform generation |
| MelGAN | GAN | Fast mel-to-waveform |
| MultiBand MelGAN | GAN | Multi-band decomposition |
| ParallelWaveGAN | GAN | Parallel waveform generation |
| UnivNet | GAN | Universal vocoder |

### Speaker Encoders

| Model | Description |
|-------|-------------|
| LSTM | LSTM-based speaker embedding extraction |
| ResNet | ResNet with squeeze-excitation attention |

## Utilities

| Tool | Description |
|------|-------------|
| `TTS/bin/synthesize.py` | CLI for speech synthesis |
| `TTS/bin/compute_embeddings.py` | Compute speaker embeddings for datasets |
| `TTS/bin/compute_statistics.py` | Compute mel-spectrogram normalization stats |
| `TTS/bin/find_unique_chars.py` | Find unique characters in a dataset |
| `TTS/bin/find_unique_phonemes.py` | Find unique phonemes in a dataset |
| `TTS/bin/remove_silence_using_vad.py` | Remove silence using VAD |
| `TTS/bin/resample.py` | Resample audio files |
| `TTS/bin/extract_tts_spectrograms.py` | Extract spectrograms with teacher forcing |

## License

This project is licensed under the [Mozilla Public License 2.0](LICENSE).
