"""Voice.Me TTS voice activity detection (VAD) utilities.

Provides functions for reading, resampling, and trimming audio files using
Silero VAD to remove silence from speech recordings.
"""

import torch
import torchaudio


def read_audio(path):
    """Read an audio file and convert to mono if multi-channel.

    Args:
        path (str): Path to the audio file.

    Returns:
        tuple: (wav, sr) where wav is a 1D tensor and sr is the sample rate.
    """
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    return wav.squeeze(0), sr


def resample_wav(wav, sr, new_sr):
    """Resample a waveform tensor to a new sample rate.

    Args:
        wav (torch.Tensor): Input waveform.
        sr (int): Original sample rate.
        new_sr (int): Target sample rate.

    Returns:
        torch.Tensor: Resampled waveform.
    """
    wav = wav.unsqueeze(0)
    transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=new_sr)
    wav = transform(wav)
    return wav.squeeze(0)


def map_timestamps_to_new_sr(vad_sr, new_sr, timestamps, just_begging_end=False):
    """Map VAD speech timestamps from one sample rate to another.

    Args:
        vad_sr (int): Sample rate used by the VAD model.
        new_sr (int): Target sample rate for the timestamps.
        timestamps (list): List of dicts with ``start`` and ``end`` keys.
        just_begging_end (bool): If True, return only the overall start/end. Defaults to False.

    Returns:
        list: Remapped timestamps as a list of dicts.
    """
    factor = new_sr / vad_sr
    new_timestamps = []
    if just_begging_end and timestamps:
        # get just the start and end timestamps
        new_dict = {"start": int(timestamps[0]["start"] * factor), "end": int(timestamps[-1]["end"] * factor)}
        new_timestamps.append(new_dict)
    else:
        for ts in timestamps:
            # map to the new SR
            new_dict = {"start": int(ts["start"] * factor), "end": int(ts["end"] * factor)}
            new_timestamps.append(new_dict)

    return new_timestamps


def get_vad_model_and_utils(use_cuda=False, use_onnx=False):
    """Load the Silero VAD model and its utility functions from torch.hub.

    Args:
        use_cuda (bool): Move model to CUDA. Defaults to False.
        use_onnx (bool): Use ONNX runtime for inference. Defaults to False.

    Returns:
        tuple: (model, get_speech_timestamps, save_audio, collect_chunks).
    """
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=use_onnx, force_onnx_cpu=True
    )
    if use_cuda:
        model = model.cuda()

    get_speech_timestamps, save_audio, _, _, collect_chunks = utils
    return model, get_speech_timestamps, save_audio, collect_chunks


def remove_silence(
    model_and_utils, audio_path, out_path, vad_sample_rate=8000, trim_just_beginning_and_end=True, use_cuda=False
):
    """Remove silence from an audio file using VAD and save the result.

    Args:
        model_and_utils (tuple): VAD model and utility functions from :func:`get_vad_model_and_utils`.
        audio_path (str): Path to the input audio file.
        out_path (str): Path to save the trimmed audio.
        vad_sample_rate (int): Sample rate for VAD processing. Defaults to 8000.
        trim_just_beginning_and_end (bool): Only trim silence at start/end. Defaults to True.
        use_cuda (bool): Move audio to CUDA for VAD. Defaults to False.

    Returns:
        tuple: (out_path, is_speech) where is_speech indicates if speech was detected.
    """
    # get the VAD model and utils functions
    model, get_speech_timestamps, _, collect_chunks = model_and_utils

    # read ground truth wav and resample the audio for the VAD
    try:
        wav, gt_sample_rate = read_audio(audio_path)
    except Exception:
        print(f"> Failed to read {audio_path}")
        return None, False

    # if needed, resample the audio for the VAD model
    if gt_sample_rate != vad_sample_rate:
        wav_vad = resample_wav(wav, gt_sample_rate, vad_sample_rate)
    else:
        wav_vad = wav

    if use_cuda:
        wav_vad = wav_vad.cuda()

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav_vad, model, sampling_rate=vad_sample_rate, window_size_samples=768)

    # map the current speech_timestamps to the sample rate of the ground truth audio
    new_speech_timestamps = map_timestamps_to_new_sr(
        vad_sample_rate, gt_sample_rate, speech_timestamps, trim_just_beginning_and_end
    )

    # if have speech timestamps else save the wav
    if new_speech_timestamps:
        wav = collect_chunks(new_speech_timestamps, wav)
        is_speech = True
    else:
        print(f"> The file {audio_path} probably does not have speech please check it !!")
        is_speech = False

    # save
    torchaudio.save(out_path, wav[None, :], gt_sample_rate)
    return out_path, is_speech
