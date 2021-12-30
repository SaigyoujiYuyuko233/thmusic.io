import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


def get_spectrogram(audio_clip, rate):
    # 频谱图
    waveform = tf.cast(audio_clip, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio_clip, nfft=1024, window=1024, stride=512)
    spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=1024, fmin=0, fmax=rate / 2)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=70)
    # spectrogram = tfio.audio.freq_mask(spectrogram, param=240)

    return spectrogram
