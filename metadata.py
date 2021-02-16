#!/usr/bin/env python3
# coding=utf-8
"""Utility functions for building a Pandas metadata table."""
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

REQUIRED_META_COLUMNS = (
    "path",
    "label",
    "split",
)


def load(corpus_dir, langs, splits):
    """Load metadata from disk into a single pandas.DataFrame.
    
    Languages are filtered based on langs.
    """
    split_dfs = []

    for split in splits:
        df = load_split(corpus_dir, langs, split)
        split_dfs.append(df)

    # Concatenate all split dataframes into a single table,
    # replace default integer indexing by utterance ids,
    # throwing an exception if there are duplicate utterance ids.
    return pd.concat(split_dfs).set_index("id", drop=True, verify_integrity=True).sort_index()


def load_split(corpus_dir, langs, split):
    """Load metadata from utt2label & utt2path to a pandas.Dataframe.
    
    Languages are filtered based on langs.
    """
    df = pd.read_csv(os.path.join(corpus_dir, split, "utt2label"), sep=" ", names=("id", "label"))
    u2p = pd.read_csv(os.path.join(corpus_dir, split, "utt2path"), sep=" ", names=("id", "path"))
    df = pd.merge(df, u2p, how="outer", on=["id"])
    df = df[df.label.isin(langs)]
    df = df.assign(split=split, corpus_dir=corpus_dir)
    return df


def load_all(data_sources, splits, num_processes=os.cpu_count()):
    """Load metadata from multiple datasets into a single table.
    
    The table has unique utterance ids for every row. 
    """
    if num_processes > 0:
        with multiprocessing.Pool(processes=num_processes) as pool:
            lang_dfs = pool.starmap(
                load, ((corpus_dir, langs, splits) for corpus_dir, langs in data_sources)
            )
    else:
        lang_dfs = (load(corpus_dir, langs, splits) for corpus_dir, langs in data_sources)
    return pd.concat(lang_dfs, verify_integrity=True).sort_index()


def verify_integrity(meta, max_threads=None):
    """Ensure that data does not have any missing values etc.

    Check that:
    1. The metadata table contains all required columns.
    2. There are no NaN values.
    3. All audio filepaths exist on disk.
    This function throws an exception if verification fails, otherwise completes silently.
    """
    missing_columns = set(REQUIRED_META_COLUMNS) - set(meta.columns)
    assert missing_columns == set(), "{} missing columns in metadata: {}".format(
        len(missing_columns), sorted(missing_columns)
    )

    assert not meta.isna().any(axis=None), "NaNs in metadata"

    if max_threads is None or max_threads > 0:
        with ThreadPoolExecutor(max_workers=max_threads) as pool:
            num_invalid = sum(
                int(not ok) for ok in pool.map(os.path.exists, meta.path, chunksize=100)
            )
    else:
        num_invalid = sum(int(not os.path.exists(path)) for path in meta.path)
    assert num_invalid == 0, "{} paths did not exist".format(num_invalid)
