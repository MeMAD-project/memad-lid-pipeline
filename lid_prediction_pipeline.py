#!/usr/bin/env python3
# coding=utf-8
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import toml
from lidbox.features import audio, cmvn
from sklearn.preprocessing import normalize

import classifier
import embedding_model as em
import feature_extraction as fe
import metadata

TF_AUTOTUNE = tf.data.experimental.AUTOTUNE
np_rng = np.random.default_rng(1)
tf.random.set_seed(np_rng.integers(0, tf.int64.max))

if len(sys.argv) < 3:
    sys.exit("Please give configuration file and data source file as arguments.")

### Since I had multiple separate scripts with, I used a separate configuration file
### to define paths in one place. Replace with hardcoded path strings if you want.
### Loading the TOML file requires the 'toml' package.
config = toml.load(sys.argv[1])
cachedir = config["paths"]["cachedir"]
modeldir = config["paths"]["modeldir"]
extractor = em.get_embedding_extractor(config["paths"]["embedding_model"])
splits = config["experiment"]["splits"]
lang_labels = config["experiment"]["langs"]
os.makedirs(cachedir, exist_ok=True)

### This bit I use to confirm that GPU is used instead of CPU. The assertion will stop the execution
### if no GPU is found. Computing embeddings with CPU is slow. Remove if not needed.

#gpus = tf.config.experimental.list_physical_devices("GPU")
#assert bool(gpus) == True, "No GPU available!"


### Read trained backend models from disk
skl_objects = classifier.pipeline_from_disk(modeldir)
scaler = skl_objects["scaler"]
dim_reducer = skl_objects["dim_reducer"]
nb_clf = skl_objects["nb_classifier"]

### These two functions are here because I am not familiar enough with Tensorflow logic to be able
### to pass the embedding model 'extractor' as an argument instead of global variable. There was
### no trivial answer available just by googling.
def batch_extract_embeddings(x):
    """Convert an audio signal to a logmel spectrogram and then into an embedding vector."""
    with tf.device("GPU"):
        #print(extractor)
        signals, rates = x["signal"], x["sample_rate"]
        S = audio.spectrograms(signals, rates[0])
        S = audio.linear_to_mel(S, rates[0])
        S = tf.math.log(S + 1e-6)
        S = cmvn(S)
        return dict(x, embedding=extractor(S))


def pipeline_from_meta(data, split):
    """Build data preprocessing pipeline."""
    to_pair = lambda x: (x["id"], x["embedding"])
    if split == "train":
        data = data.sample(frac=1, random_state=np_rng.bit_generator)

    print("Read audio in...")
    ds = tf.data.Dataset.from_tensor_slices(fe.metadata_to_dataset_input(data)).map(
        fe.read_audio, num_parallel_calls=TF_AUTOTUNE
    )

    print("Extract embeddings...")
    if split == "train":
        ds = (
            ds.prefetch(1000)
            .apply(fe.create_chunks)
            .batch(128)
            .map(batch_extract_embeddings, num_parallel_calls=TF_AUTOTUNE)
            .unbatch()
            .map(to_pair, num_parallel_calls=TF_AUTOTUNE)
        )
    else:
        ds = (
            ds.apply(fe.create_chunks)
            .batch(128)
            .map(batch_extract_embeddings, num_parallel_calls=TF_AUTOTUNE)
            .unbatch()
            .map(to_pair, num_parallel_calls=TF_AUTOTUNE)
            .prefetch(1000)
        )
    ids = []
    embeddings = []
    
    for i, embedding in ds.as_numpy_iterator():
        ids.append(i.decode("utf-8"))
        embeddings.append(embedding.astype(np.float32))

    print("Returning embeddings.")
    df = fe.embeddings_to_dataframe(ids, embeddings)
    return df


### Load data using paths in the supplied file
with open(sys.argv[2], "r") as infile:
    data_sources = infile.readlines()

corpora = []
for source in data_sources:
    corpora.append((source.strip(), lang_labels))

meta = metadata.load_all(corpora, splits)
meta, lang2target = classifier.generate_lang2target(meta, lang_labels)

print("\nsize of all metadata", meta.shape)
meta = meta.dropna()
print("after dropping NaN rows", meta.shape)
print("verifying integrity")
metadata.verify_integrity(meta)
print("ok\n")

### Compute embeddings
embs = pipeline_from_meta(meta, splits[0])

### Copy metadata to embedding table (embedding table contains the data chunks)
get_parent_id = lambda t: t.rsplit("-", 1)[0]
embs["group"] = embs.sort_index().groupby(get_parent_id).ngroup()
meta["group"] = meta.groupby("id").ngroup()
embs = embs.reset_index().merge(meta, how="left", on="group").set_index("id")
assert not embs.embedding.isna().any(
    axis=None
), "Missing embeddings, some rows contained NaN values"

### Extract embeddings as numpy arrays for the sklearn models
data_X, data_y = classifier.embeddings_as_numpy_data(embs)
print("prediction vectors", data_X.shape, data_y.shape)

### Standardize all vectors using training set statistics
print("Predict first with Naive Bayes model.")
data_X = scaler.transform(data_X)

### Use PLDA to reduce dimensions
data_X = dim_reducer.transform(data_X)

### L2-normalize vectors to surface of a unit sphere
data_X = normalize(data_X)

### Predict scores on test set with classifier and compute metrics
predictions = nb_clf.predict_log_proba(data_X)
### Clamp -infs to -100
predictions = np.maximum(-100, predictions)

### Predictions are computed for 2 second chunks, now calculate the average of the chunks
### for each input audio
def average_chunk_predictions(rows):
    if len(rows) == 1:
        return np.stack(rows).argmax()
    return np.stack(rows).mean(axis=0).argmax()


chunk_ids = embs.index.tolist()
result = pd.DataFrame.from_dict({"id": chunk_ids, "nb_pred": predictions.tolist()}).set_index("id")
result = result.sort_index().groupby(get_parent_id).agg({"nb_pred": average_chunk_predictions})
result = result.assign(corpus_dir=meta.corpus_dir)

### Write result to an 'utt2lang' file where each utterance id is paired with the predicted
### language for that utterance id 
target2lang = {target: lang for lang, target in lang2target.items()}
result.nb_pred = result.nb_pred.apply(lambda x: target2lang[x])

for ids, corpus in result.groupby("corpus_dir"):
    cdir = corpus.corpus_dir[0]
    print(f"Writing language predictions for {cdir} to an utt2lang file.")
    #corpus["nb_pred"].to_csv(f"{cdir}/{splits[0]}/utt2lang", sep=" ", header=False)
    corpus["nb_pred"].to_json(f"{cdir}/{splits[0]}/utt2lang.json")

