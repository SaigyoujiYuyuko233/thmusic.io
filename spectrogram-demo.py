import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt
from util import audio

CLIP_SIZE = 240000
if __name__ == '__main__':
    # Read
    rate, audio_mono = audio.parse_audio("./sakurasakura-demo/dataset-compressed/prediction/sakurasakura - 狐の工作室.mp3")

    # 转换为单声道
    # audio_mono = tf.convert_to_tensor(np.sum(audio[:].numpy(), axis=1) / 2.0, dtype=tf.dtypes.float32)
    # audio_mono = tf.reshape(audio_mono, shape=[len(audio_mono), 1])
    # audio_mono = tf.squeeze(audio_mono, axis=[-1])

    # 片段
    audio_mid = int(len(audio_mono) / 2)
    clip_mid = int(CLIP_SIZE / 2)
    audio_clip = audio_mono[audio_mid - clip_mid: audio_mid + clip_mid]

    # audio_enc = tfio.audio.encode_mp3(tf.reshape(audio_clip, shape=[len(audio_clip), 1]), rate=rate)
    # tf.io.write_file("processed.mp3", audio_enc)

    # 频谱图
    waveform = tf.cast(audio_clip, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio_clip, nfft=2048, window=800, stride=1024)
    spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=240, fmin=0, fmax=rate / 2)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=20)
    spectrogram = spectrogram.numpy()

    # mask
    for i in range(0, 20):
        spectrogram[:, i] = np.zeros(len(spectrogram))
    spectrogram = tf.convert_to_tensor(spectrogram)

    plt.imshow(tf.math.log(spectrogram).numpy())
    plt.show()
