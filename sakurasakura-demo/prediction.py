from util import sample
import tensorflow as tf
import glob
import numpy as np

MODEL = "./models/model_1.0_2021_12_31-20_03_33.h5"
CLIP_SIZE = 40000

if __name__ == '__main__':
    model = tf.keras.models.load_model(MODEL)
    model.summary()

    for file in glob.glob("./dataset-compressed/prediction/*.mp3"):
        spects, _ = sample.gen_train_sample(file, label=1, clip_size=CLIP_SIZE)
        results = model.predict(tf.reshape(spects, [-1, spects[0].shape[0], spects[0].shape[1], 1]))
        results = results[np.logical_not(np.isnan(results))]
        result = np.sum(results) / len(results)
        print("[{file}]: {result}".format(file=file, result=result))
