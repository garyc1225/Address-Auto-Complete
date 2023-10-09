import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import einops
import random

def order_random_sample(myList, K, lb=2):
    """This function takes a list, and randomly sample K components from the list in a sorted order

    Args:
        K: (int). Number of components to pick

    Returns:
        List
    """
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

    # Create the final string
    final_columns = ['AddressNumber','StreetName','PostType','ZipName','CountyName','State','ZipCode']

    context_df = df.sample(target, replace=True)[final_columns].copy()
    target_df = context_df.copy()

    context_df['final_str_l'] = context_df.astype(str).apply(lambda x:','.join(x), axis = 1).apply(lambda x:x.split(',')).apply(lambda x:order_random_sample(x,3))
    context_df['final_str'] = context_df['final_str_l'].apply(lambda x:' '.join(filter(None, x)))
    target_df['final_str'] = target_df.astype(str).apply(lambda x:' '.join(filter(None, x)), axis = 1)

    return np.array(context_df['final_str'].tolist()),np.array(target_df['final_str'].tolist())

def tf_lower_and_split_punct(text):
    """This function takes a string, and returns the standardized string with Start of String (SOS) and End of String (EOS) token.
    """
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

def top_5_index(tensor, k = 5):
    """Returns the index with the top 5 largest values of a tensor.

    Args:
        tensor: A tensor.

    Returns:
        A tensor of the index with the top 5 largest values of the tensor.
    """

    __= tf.math.top_k(tensor, k)
    return __.indices.numpy()