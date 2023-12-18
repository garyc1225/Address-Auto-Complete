import pandas as pd
import numpy as np
import utils
import seaborn as sns
import tensorflow as tf
import tensorflow_text as tf_text

UNITS = 256

class GRU_Encoder(tf.keras.layers.Layer):
    """An encoder model that encodes input sequences into hidden states."""

    def __init__(self, text_processor, units=UNITS):
        """Initializes the encoder.

        Args:
            text_processor: A text processor that can be used to convert text sequences to tensors and vice versa.
            units: The dimension of the hidden states.
        """

        super(GRU_Encoder, self).__init__()

        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)

        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(units, return_sequences=True, recurrent_initializer='glorot_uniform'))

    def call(self, x):
        """Encodes the input sequences.

        Args:
            x: A tensor of shape `[batch_size, max_sequence_length]`.

        Returns:
            A tensor of shape `[batch_size, max_sequence_length, units]`.
        """

        # Check the shape of the input tensor.
        shape_checker = utils.ShapeChecker()
        shape_checker(x, 'batch s')

        # Embed the input sequences.
        x = self.embedding(x)

        # Check the shape of the embedded input sequences.
        shape_checker(x, 'batch s units')

        # Encode the input sequences using the bidirectional LSTM layer.
        x = self.rnn(x)

        # Check the shape of the encoded input sequences.
        shape_checker(x, 'batch s units')

        return x

    def convert_input(self, texts):
        """Converts the input texts to a tensor of hidden states.

        Args:
            texts: A list of strings.

        Returns:
            A tensor of shape `[batch_size, max_sequence_length, units]`.
        """

        # Convert the input texts to a tensor of text sequences.
        texts = tf.convert_to_tensor(texts)

        # If the input tensor is a scalar, convert it to a vector.
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]

        # Convert the text sequences to a tensor of hidden states.
        context = self.text_processor(texts).to_tensor()
        context = self(context)

        return context

class CrossAttention(tf.keras.layers.Layer):
    """A cross-attention layer that attends to a context sequence to compute a new representation of an input sequence.

    Args:
        units: The dimension of the hidden states.
        kwargs: Additional arguments to pass to the MultiHeadAttention layer.
    """

    def __init__(self, units, **kwargs):
        super().__init__()

        # Create a MultiHeadAttention layer.
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)

        # Create a LayerNormalization layer.
        self.layernorm = tf.keras.layers.LayerNormalization()

        # Create an Add layer.
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        """Computes the cross-attention output.

        Args:
            x: A tensor of shape `[batch_size, input_sequence_length, units]`.
            context: A tensor of shape `[batch_size, context_sequence_length, units]`.

        Returns:
            A tensor of shape `[batch_size, input_sequence_length, units]`.
        """

        # Check the shape of the input tensors.
        shape_checker = utils.ShapeChecker()
        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')

        # Compute the attention output and attention scores.
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)

        # Check the shape of the attention output and attention scores.
        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')

        # Reduce the attention scores to a single head.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)

        # Check the shape of the reduced attention scores.
        shape_checker(attn_scores, 'batch t s')

        # Store the attention weights for later use.
        self.last_attention_weights = attn_scores

        # Compute the new representation of the input sequence.
        x = self.add([x, attn_output])

        # Normalize the new representation of the input sequence.
        x = self.layernorm(x)

        return x

    def get_shape(self):
        """Prints the shape of the weights of the MultiHeadAttention layer."""
        weight_names = ['query', 'keys', 'values', 'proj']
        for name, out in zip(weight_names,self.mha.get_weights()):
            print(name, out.shape)

class GRU_Decoder(tf.keras.layers.Layer):
    """This class defines a decoder for a sequence-to-sequence model. It uses an LSTM to generate the next token in the sequence, and an attention mechanism to attend to the encoder output.

    Args:
        text_processor: A TextProcessor object.
        units: The number of units in the LSTM layer.
    """

    def __init__(self, text_processor, units):
        super(GRU_Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')

        self.units = units

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        # The embedding layer converts the input tokens into dense vectors.

        self.rnn = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        # The GRU layer generates the next token in the sequence, based on the previous tokens and the encoder output.

        self.attention = CrossAttention(units)
        # The attention layer attends to the encoder output, and uses the attended information to generate the next token.

        self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation = 'sigmoid')
        # The output layer converts the dense vectors generated by the GRU layer into probabilities over the next token.

    def call(self,
            context, x,
            state=None,
            return_state=False):  
        """
            Decodes a sequence of tokens based on the context.

            Args:
              context: A tensor of shape (batch_size, encoder_seq_len, encoder_units).
              x: A tensor of shape (batch_size, target_seq_len).
              state: A tuple of tensors, (h_state, c_state), of shape (batch_size, units).
              return_state: Whether to return the state of the decoder.

            Returns:
              A tensor of shape (batch_size, target_seq_len, target_vocab_size) containing the logits for the next token in the sequence.
            """

        shape_checker = utils.ShapeChecker()
        shape_checker(x, 'batch t')
        shape_checker(context, 'batch s units')

        # 1. Lookup the embeddings
        x = self.embedding(x)
        shape_checker(x, 'batch t units')

        # 2. Process the target sequence.
        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, 'batch t units')

        # 3. Use the RNN output as the query for the attention over the context.
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, 'batch t units')
        shape_checker(self.last_attention_weights, 'batch t s')

        # Step 4. Generate logit predictions for the next token.
        logits = self.output_layer(x)
        shape_checker(logits, 'batch t target_vocab_size')

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=' ')
        result = tf.strings.regex_replace(result, '^ *\[START\] *', '')
        result = tf.strings.regex_replace(result, ' *\[END\] *$', '')
        return result

    def get_next_token(self, context, next_token, done, state, random = False):
        logits, state = self(
            context, next_token,
            state = state,
            return_state=True) 

        if not random:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :]/1
            next_token = tf.random.categorical(logits, num_samples=1)

        # If a sequence produces an `end_token`, set it `done`
        done = done | (next_token == self.end_token)
        # Once a sequence is done it only produces 0-padding.
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state, logits

class Addressor(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units,
                context_text_processor,
                target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = GRU_Encoder(context_text_processor, units)
        decoder = GRU_Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        return logits

def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)

def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

@Addressor.add_method
def Addressor_fix(self,
                  texts, *,
                  max_length=50,
                  random=False):
    # Process the input texts
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]

    # Setup the loop inputs
    tokens = []
    attention_weights = []
    logits_l = []
    next_token, done, state = self.decoder.get_initial_state(context)

    for _ in range(max_length):
        # Generate the next token
        next_token, done, state, logits = self.decoder.get_next_token(
            context, next_token, done,  state, random)

        # Collect the generated tokens
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)
        logits_l.append(logits)

        if tf.executing_eagerly() and tf.reduce_all(done):
            break

    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1)   # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)
    self.all_logis = tf.concat(logits_l, axis=1)

    result = self.decoder.tokens_to_text(tokens)
    return result

@Addressor.add_method
def plot_attention(self, text, **kwargs):
    assert isinstance(text, str)
    output = self.Addressor_fix([text], **kwargs)
    output = output[0].numpy().decode()

    attention = self.last_attention_weights[0]

    context = utils.tf_lower_and_split_punct(text)
    context = context.numpy().decode().split()

    output = utils.tf_lower_and_split_punct(output)
    output = output.numpy().decode().split()[1:]

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(1, 1, 1)

    ax = sns.heatmap(model.last_attention_weights[0], annot=True, fmt=".3f")

    fontdict = {'fontsize': 14}

    ax.set_xticklabels(context, fontdict=fontdict)
    ax.set_yticklabels(output, fontdict=fontdict)

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')