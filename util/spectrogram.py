import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


def get_spectrogram(audio_clip, rate):
    # 频谱图
    waveform = tf.cast(audio_clip, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio_clip, nfft=2048, window=800, stride=1024)
    spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=220, fmin=0, fmax=rate / 2)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=28)

    # mask
    spectrogram = np.delete(spectrogram, range(0, 10), axis=1)
    for i in range(0, 24):
        spectrogram[:, i] = np.zeros(len(spectrogram))
    spectrogram = tf.convert_to_tensor(spectrogram)

    return spectrogram
