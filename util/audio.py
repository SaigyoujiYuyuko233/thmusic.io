import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import pydub


def parse_audio(file):
    # Read
    audio = pydub.AudioSegment.from_mp3(file)
    audio_mono = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        audio_mono = audio_mono.reshape((-1, 2))
    audio_mono = np.float32(audio_mono) / 2 ** 15

    # 转换为单声道
    audio_mono = tf.convert_to_tensor(np.sum(audio_mono, axis=1) / 2.0, dtype=tf.dtypes.float32)
    audio_mono = tf.reshape(audio_mono, shape=[len(audio_mono), 1])
    audio_mono = tf.squeeze(audio_mono, axis=[-1])

    return audio.frame_rate, audio_mono
