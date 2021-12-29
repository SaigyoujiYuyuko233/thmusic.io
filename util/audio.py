import tensorflow as tf
import tensorflow_io as tfio
import numpy as np


def parse_audio(file):
    # Read
    audio = tfio.audio.AudioIOTensor(file, dtype=tf.dtypes.float32)

    # 转换为单声道
    audio_mono = tf.convert_to_tensor(np.sum(audio[:].numpy(), axis=1) / 2.0, dtype=tf.dtypes.float32)
    audio_mono = tf.reshape(audio_mono, shape=[len(audio_mono), 1])
    audio_mono = tf.squeeze(audio_mono, axis=[-1])

    return audio, audio_mono
