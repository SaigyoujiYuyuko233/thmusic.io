import pydub
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import matplotlib.pyplot as plt

CLIP_SIZE = 800000
if __name__ == '__main__':
    # Read
    audio = pydub.AudioSegment.from_mp3("./sakurasakura-demo/dataset-compressed/test/sakurasakura/sample-2.mp3")
    audio_mono = np.array(audio.get_array_of_samples())
    if audio.channels == 2:
        audio_mono = audio_mono.reshape((-1, 2))
    audio_mono = np.float32(audio_mono) / 2 ** 15

    audio_mono = tf.reshape(audio_mono, shape=[len(audio_mono), 1])
    audio_mono = tf.squeeze(audio_mono, axis=[-1])

    # 片段
    audio_mid = int(len(audio_mono) / 2)
    clip_mid = int(CLIP_SIZE / 2)
    audio_clip = audio_mono[audio_mid - clip_mid: audio_mid + clip_mid]

    # audio_enc = tfio.util.encode_mp3(tf.reshape(audio_clip, shape=[len(audio_clip), 1]), rate=util.rate.numpy())
    # tf.io.write_file("processed.mp3", audio_enc)

    # 频谱图
    waveform = tf.cast(audio_clip, tf.float32)
    spectrogram = tfio.audio.spectrogram(audio_clip, nfft=1024, window=1024, stride=512)
    spectrogram = tfio.audio.melscale(spectrogram, rate=audio.frame_rate, mels=1024
                                      , fmin=0, fmax=audio.frame_rate / 2)
    spectrogram = tfio.audio.dbscale(spectrogram, top_db=70)
    # spectrogram = tfio.audio.freq_mask(spectrogram, param=280)

    plt.imshow(spectrogram.numpy())
    plt.show()

