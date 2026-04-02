import keras
import tensorflow as tf
from keras.src.metrics import AUC
from keras.src.layers import LayerNormalization
from keras.src.layers import Layer

from keras.src.layers import Lambda, Conv1D, Concatenate, BatchNormalization, Activation, GlobalMaxPooling1D
from tensorflow.keras.models import load_model

"""CHANGES:
1. metrics is changed to AUC ROC and AUC PR
2. added LayerNormalization
3. added Reverse complementation layer (unused)
4. added extract_channel.
5. created G4StackConv: a Conv1D layer on only raw G line alone and concated to the data
6. No RCConv1D is used.
7. Added a c_raw channel as well to G4Conv
"""


def extract_channel(x_lc, idx=3):
    """
    x_lc: (B, L, C) channels-last tensor
    order: e.g., 'ATCGP' for [A,T,C,G,P]
    which: one of 'A','T','C','G','P'
    returns (B, L, 1)
    """
    return x_lc[:, :, idx:idx + 1]

@keras.saving.register_keras_serializable()
class G4StackConv1D(keras.layers.Layer):
    """
    Reverse-Complement equivariant Conv1D:
      - Apply the SAME Conv1D to x and RC(x)
      - Pool their responses by max (you can switch to mean if you prefer)

    Args:
      filters, kernel_size: passed to Conv1D
      swap_channels: True for the *first* RC layer on raw bases; False for feature maps
      swapped_order: base channel order, e.g. "ATCGP" for [A,T,C,G,PairProb]
      **kwargs: any Conv1D kwargs (activation, kernel_regularizer, etc.)
    """

    def __init__(self, filters, kernel_size, g_channel_idx, c_channel_idx, **kwargs):
        super().__init__()

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.g_channel_idx = int(g_channel_idx)
        self.c_channel_idx = int(c_channel_idx)

        # Save Conv1D kwargs for serialization
        self.conv_kwargs = dict(kwargs)
        # We create the conv in build() so it’s compatible with serialization

    def build(self, input_shape):
        # Force channels_last for clarity
        self.conv = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same", use_bias=True,
            #data_format="channels_last",
            **self.conv_kwargs
        )
        super().build(input_shape)

    def call(self, x):
        g_raw = extract_channel(x, idx=self.g_channel_idx)
        c_raw = extract_channel(x, idx=self.c_channel_idx)
        g_feats=[
            self.conv(g_raw),
            self.conv(c_raw)
        ]
        return Concatenate(axis=-1,name="G4StackConv")(g_feats)
        #return tf.maximum(*g_feats)

        # y_rc = self.conv(rc_tensor(x, swap_channels=self.swap_channels, order=self.swapped_order))
        # y_fwd = self.conv(tf.gather(x, indices=self.unswapped_order, axis=2) if self.swap_channels else x)
        # # y_fwd=  tf.gather(x, indices=self.unswapped_order, axis=2) if self.swap_channels else x
        # # y_rc=  rc_tensor(x, swap_channels=self.swap_channels, order=self.swapped_order)
        #
        # # y_fwd = self.conv(x)
        # # y_rc  = self.conv(rc_tensor(x, swap_channels=self.swap_channels, order=self.order))
        # return tf.maximum(y_fwd, y_rc)

    # --- Serialization ---
    def get_config(self):
        base = super().get_config()
        base.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "g_channel_idx": self.g_channel_idx,
            "c_channel_idx": self.c_channel_idx,
            **self.conv_kwargs,  # include activation, regularizers, etc.
        })
        return base

    @classmethod
    def from_config(cls, config):
        # Keras will pass the merged dict; pop our known args, rest go to Conv1D kwargs
        filters = config.pop("filters")
        kernel_size = config.pop("kernel_size")
        c_channel_idx = config.pop("c_channel_idx", 2)
        g_channel_idx = config.pop("g_channel_idx", 3)

        return cls(filters, kernel_size, g_channel_idx=g_channel_idx, c_channel_idx=c_channel_idx, **config)



def rc_tensor(x, swap_channels: bool = True, order: tuple[int] = (1, 0, 3, 2)):
    """
    Channels-last RC: reverse along sequence (axis=1).
    If swap_channels=True, swap base channels per `order`:
      order can be tuple of original channels indices, etc.
      ie. (1,0,3,2,4) implying 1 and 0 replace, 3 and 2 replace, 4 remains same
    """
    x_rev = tf.reverse(x, axis=[1])  # reverse along length
    if not swap_channels:
        return x_rev

    return tf.gather(x_rev, indices=order, axis=2)

@keras.saving.register_keras_serializable()
class RCConv1D(keras.layers.Layer):
    """
    Reverse-Complement equivariant Conv1D:
      - Apply the SAME Conv1D to x and RC(x)
      - Pool their responses by max (you can switch to mean if you prefer)

    Args:
      filters, kernel_size: passed to Conv1D
      swap_channels: True for the *first* RC layer on raw bases; False for feature maps
      swapped_order: base channel order, e.g. "ATCGP" for [A,T,C,G,PairProb]
      **kwargs: any Conv1D kwargs (activation, kernel_regularizer, etc.)
    """

    def __init__(self, filters, kernel_size, swap_channels=True, unswapped_order=(0, 1, 2, 3,4,5),
                 swapped_order=(1, 0, 3, 2,5,4), **kwargs):
        super().__init__()
        # Validate: order is a permutation of 0..n_ch-1
        tf.debugging.assert_equal(
            tf.shape(unswapped_order)[0], tf.shape(swapped_order)[0],
            message="`swapped and unswapped order must be of equal length"
        )

        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.swap_channels = bool(swap_channels)
        self.swapped_order = swapped_order
        self.unswapped_order = unswapped_order
        # Save Conv1D kwargs for serialization
        self.conv_kwargs = dict(kwargs)
        # We create the conv in build() so it’s compatible with serialization

    def build(self, input_shape):
        # Force channels_last for clarity
        self.conv = keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding="same",
            data_format="channels_last",
            **self.conv_kwargs
        )
        super().build(input_shape)

    def call(self, x):
        if (self.swap_channels):
            tf.debugging.assert_less(
                max(self.unswapped_order), tf.shape(x)[-1],
                message="`max index number must be less than the channel count"
            )

        y_rc = self.conv(rc_tensor(x, swap_channels=self.swap_channels, order=self.swapped_order))
        y_fwd = self.conv(tf.gather(x, indices=self.unswapped_order, axis=2) if self.swap_channels else x)
        # y_fwd=  tf.gather(x, indices=self.unswapped_order, axis=2) if self.swap_channels else x
        # y_rc=  rc_tensor(x, swap_channels=self.swap_channels, order=self.swapped_order)

        # y_fwd = self.conv(x)
        # y_rc  = self.conv(rc_tensor(x, swap_channels=self.swap_channels, order=self.order))
        return tf.maximum(y_fwd, y_rc)

    # --- Serialization ---
    def get_config(self):
        base = super().get_config()
        base.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "swap_channels": self.swap_channels,
            "swapped_order": self.swapped_order,
            "unswapped_order": self.unswapped_order,
            **self.conv_kwargs,  # include activation, regularizers, etc.
        })
        return base

    @classmethod
    def from_config(cls, config):
        # Keras will pass the merged dict; pop our known args, rest go to Conv1D kwargs
        filters = config.pop("filters")
        kernel_size = config.pop("kernel_size")
        swap_channels = config.pop("swap_channels", True)
        # unswapped_order = config.pop("unswapped_order", (0, 1, 2, 3))
        unswapped_order = config.pop("unswapped_order", (0, 1, 2, 3,4,5))
        # swapped_order = config.pop("swapped_order", (1, 0, 3, 2))
        swapped_order = config.pop("swapped_order", (1, 0, 3, 2,5,4))

        return cls(filters, kernel_size, swap_channels=swap_channels, unswapped_order=unswapped_order,
                   swapped_order=swapped_order, **config)


def build_conv_block(input_layer, filter_size, kernel_sizes, is_concatenate=True, use_layer_norm=True):
    """
    This function builds a multi-scale convolution block with batch normalization and ELU activations.
    Each convolutional layer uses a different kernel size.
    """
    assert len(filter_size) == len(kernel_sizes), "filter sizes and kernel sizes must be of equal length"
    conv_layers = []
    last_conv_layer = input_layer
    for i, filter_count in enumerate(filter_size):
        # Build each convolutional layer
        conv_layer = Conv1D(
            filters=filter_count,
            kernel_size=kernel_sizes[i],
            strides=1,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(5e-7),
            padding='same',
            use_bias=True,
            bias_initializer='RandomNormal'
        )(input_layer if is_concatenate else last_conv_layer)
        # Batch Normalization and Activation
        conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)
        conv_layer = tf.keras.layers.Activation("elu")(conv_layer)
        conv_layers.append(conv_layer)
        last_conv_layer = conv_layer

    if use_layer_norm:
        pooled_layers = [tf.keras.layers.GlobalMaxPooling1D()(LayerNormalization(axis=-1, epsilon=1e-5)(conv_layer)) for
                         conv_layer in conv_layers]
    else:
        # Apply Global Max Pooling to each convolutional output
        pooled_layers = [tf.keras.layers.GlobalMaxPooling1D()(conv_layer) for conv_layer in conv_layers]

    # Concatenate pooled outputs from all convolutions
    if is_concatenate:
        concatenated_output = tf.keras.layers.Concatenate(axis=-1)(pooled_layers)

    return concatenated_output if is_concatenate else tf.keras.layers.GlobalMaxPooling1D()(last_conv_layer)

def make_rc_orders(n_raw_channels: int, n_total_channels: int):
    # raw channels are [A,T,C,G] (+ optional P at index 4)
    # extra channels (e.g. from G4 stack) are assumed to come in swap-pairs
    unswapped = tuple(range(n_total_channels))

    swapped = [1, 0, 3, 2]  # swap A<->T, C<->G

    if n_raw_channels > 4:
        # leave P (unpairProb) unchanged at index 4
        swapped.append(4)

    start = n_raw_channels
    for i in range(start, n_total_channels, 2):
        if i + 1 < n_total_channels:
            swapped.extend([i + 1, i])  # swap each pair
        else:
            swapped.append(i)  # odd leftover (shouldn't happen here, but safe)

    return unswapped, tuple(swapped)


def build_model(input_shape, lr, opt_func, dense_units,
                applyConvBlock=True, filter_sizes=(64, 80), kernel_sizes=(5, 7), is_concatenated: bool = True,
                applyG4Stack: bool = True, g_channel_idx=3, c_channel_idx=2, g_feature_kernels=(3, 5, 7),
                applyRCConv: bool = True, RC_filters=(64, 64), RC_kernels=(7, 5)
                ):
    """
    This function builds a model with a multi-scale convolutional block followed by dense layers.
    The output layer uses a sigmoid activation for binary classification.
    """
    print(input_shape)

    input_layer = tf.keras.layers.Input(shape=input_shape)

    # input_layer: (B, 5, 100) -> permute to (B, 100, 5) once
    x = keras.layers.Permute((2, 1), name="to_channels_last")(input_layer)

    if applyG4Stack:
        # # # #GStackConv
        g_feats = [
            G4StackConv1D(
                filters=1, kernel_size=k,
                g_channel_idx=g_channel_idx, c_channel_idx=c_channel_idx,
                name=f"G4_Stack_Conv_k{k}")(x)
                      for k in g_feature_kernels
                  ]

        # 3) Stack them along the channel axis -> (B, len(kernel_sizes), L)
        g_stack = Concatenate(axis=-1, name="g_run_stack")(g_feats)
        g_stack = BatchNormalization(axis=1, name="g_run_bn")(g_stack)
        g_stack = Activation("relu", name="g_run_relu")(g_stack)

        x = keras.layers.Concatenate(axis=-1, name="concat_glearn")([x, g_stack])
        # # #END GStack Conv


    # if applyRCConv:
    #     # RC on raw channels (swap bases)
    #     x = RCConv1D(
    #         RC_filters[0], RC_kernels[0], swap_channels=True,
    #         unswapped_order=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    #         swapped_order=(1, 0, 3, 2, 5, 4, 7, 6, 9, 8),
    #         kernel_initializer="he_normal",
    #         kernel_regularizer=keras.regularizers.l2(5e-7),
    #         use_bias=True
    #     )(x)
    if applyRCConv:
        n_raw = input_shape[0]  # 4 or 5 (if includeUnpairProb)
        n_extra = (2 * len(g_feature_kernels)) if applyG4Stack else 0
        n_total = n_raw + n_extra

        unswapped_order, swapped_order = make_rc_orders(n_raw, n_total)

        x = RCConv1D(
            RC_filters[0], RC_kernels[0], swap_channels=True,
            unswapped_order=unswapped_order,
            swapped_order=swapped_order,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(5e-7),
            use_bias=True
        )(x)

        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.Activation("elu")(x)

        # RC on feature maps (no channel swap)
        x = RCConv1D(
            RC_filters[1], RC_kernels[1], swap_channels=False,
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(5e-7),
            use_bias=True
        )(x)
        x = keras.layers.BatchNormalization(axis=-1)(x)
        x = keras.layers.Activation("elu")(x)


    # Now feed RC-features into your existing conv/multiscale block
    if applyConvBlock:
        feature_layer = build_conv_block(
            x,  # <— previously input_layer
            filter_sizes, kernel_sizes,
            is_concatenate=is_concatenated
        )
        x = feature_layer
    else:
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

    # Fully connected layers
    for units in dense_units:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.Activation("elu")(x)
        # x = tf.keras.layers.Dropout(0.2)(x)  # Dropout to prevent overfitting 0.2 > 0.5

    # Output layer with sigmoid activation for binary classification
    output_layer = tf.keras.layers.Dense(1)(x)
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer)

    # Build the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    # Select optimizer
    if 'sgd' in opt_func:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr, decay=1e-4, momentum=0.99, nesterov=True)
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=1e-5)

    # Compile the model
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[AUC(name='auc', curve='ROC'),
                           AUC(name='auprc', curve='PR')])
    return model


def loadModel(modelFilepath):
    return load_model(modelFilepath, custom_objects={"RCConv1D": RCConv1D, "rc_tensor": rc_tensor,
                                                     "G4StackConv1D":G4StackConv1D,"extract_channel":extract_channel})
