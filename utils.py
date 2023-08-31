import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import einops
import random

def order_random_sample(myList,K,lb=2):
    indices = random.sample(range(len(myList)-lb), K)
    indices = [x+lb for x in indices]
    for i in range(lb):
        indices.append(i)
    return [myList[i] for i in sorted(indices)]

def create_label_target(loc, ignore_columns = [], target = 2000000):
    """This function takes the raw Address dataset, and then output final df with training label.

    Args:
        loc: (str). File Location for the raw Address dataset

    Returns:
        context_raw: (list) List of Address, with masking
        target_raw: (list) List of Address, without masking
    """

    df = pd.read_parquet(loc, engine='pyarrow')
    df = df.loc[:,lambda x:~x.columns.isin(ignore_columns)].copy()
    df['ZipCode'] = df['ZipCode'].astype(int)

    context_df = df.sample(target, replace=True).copy()
    target_df = context_df.copy()

    # context_df['rand'] = np.random.uniform(0,1,context_df.shape[0])

    # # Mask the word according to probability
    # context_df.loc[lambda x:x['rand'].between(0,0.1, inclusive = 'left'), 'AddressNumber'] = ''
    # context_df.loc[lambda x:x['rand'].between(0.1,0.2, inclusive = 'left'), 'StreetName'] = ''
    # context_df.loc[lambda x:x['rand'].between(0.2,0.3, inclusive = 'left'), 'PostType'] = ''
    # context_df.loc[lambda x:x['rand'].between(0.4,0.7, inclusive = 'left'), 'ZipCode'] = ''
    # # context_df.loc[lambda x:x['rand'].between(0.8,0.9, inclusive = 'left'), 'CountyName'] = ''
    # context_df.loc[lambda x:x['rand'].between(0.7,1, inclusive = 'left'), 'ZipName'] = ''

    # Create the final string
    final_columns = ['AddressNumber','StreetName','PostType','ZipName','CountyName','State','ZipCode']

    context_df['final_str_l'] = context_df[final_columns].astype(str).apply(lambda x:','.join(x), axis = 1).apply(lambda x:x.split(',')).apply(lambda x:order_random_sample(x,3))
    context_df['final_str'] = context_df['final_str_l'].apply(lambda x:' '.join(filter(None, x)))
    target_df['final_str'] = target_df[final_columns].astype(str).apply(lambda x:' '.join(filter(None, x)), axis = 1)

    return np.array(context_df['final_str'].tolist()),np.array(target_df['final_str'].tolist())

def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.,0-9]', '')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                f"    found: {new_dim}\n"
                                f"    expected: {old_dim}\n")