import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

CLIP_SIZE = 480000
if __name__ == '__main__':
    # Read
    audio = tfio.audio.AudioIOTensor("./demo-data/train/sakurasakura/sakurasakura-origion.mp3", dtype=tf.dtypes.float32)

    # 转换为单声道
    audio_mono = tf.convert_to_tensor(np.sum(audio[:].numpy(), axis=1) / 2.0, dtype=tf.dtypes.float32)
    audio_mono = tf.reshape(audio_mono, shape=[len(audio_mono), 1])
    audio_mono = tf.squeeze(audio_mono, axis=[-1])

    # 片段
    audio_mid = int(len(audio_mono) / 2)
    clip_mid = int(CLIP_SIZE / 2)
    audio_clip = audio_mono[audio_mid - clip_mid: audio_mid + clip_mid]

    audio_enc = tfio.audio.encode_mp3(tf.reshape(audio_clip, shape=[len(audio_clip), 1]), rate=audio.rate.numpy())
    tf.io.write_file("processed.mp3", audio_enc)

    # 频谱图
    waveform = tf.cast(audio_clip, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio_clip, nfft=1024, window=512, stride=256)
    spectrogram = tfio.audio.melscale(spectrogram, rate=audio.rate.numpy(), mels=1024, fmin=0, fmax=audio.rate.numpy() / 2)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=50)
    spectrogram = tfio.audio.freq_mask(spectrogram, param=240)

    plt.imshow(spectrogram.numpy())
    plt.show()
