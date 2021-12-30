import util.audio

from util import spectrogram
import matplotlib.pyplot as plt


def gen_train_sample(file, label, clip_size):
    rate, audio_mono = util.audio.parse_audio(file)
    clips = int(audio_mono.shape[0] / clip_size)

    spect = []
    labels = []
    for seg in range(0, clips):
        spect.append(spectrogram.get_spectrogram(audio_mono[seg * clip_size: seg * clip_size + clip_size], rate))
        labels.append(label)

    return spect, labels
