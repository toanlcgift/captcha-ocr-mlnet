# captcha-ocr-mlnet
Implementation of Captcha OCR using CNNs, RNNs and CTC loss

.net implementation of https://github.com/keras-team/keras-io/blob/master/examples/vision/captcha_ocr.py

<p align="center">
<img src="docs/console.png" alt="console output" height="300" >
<img src="ConsoleApp/test.png" alt="raw image" height="50" >
</p>

Note: can use following code to convert from keras model to onnx model
``` python
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime
import keras
from keras import ops
from keras import layers
from tensorflow.keras.models import load_model

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, axis=-1), dtype="int32")
    input_length = ops.cast(ops.squeeze(input_length, axis=-1), dtype="int32")
    sparse_labels = ops.cast(
        ctc_label_dense_to_sparse(y_true, label_length), dtype="int32"
    )

    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())

    return ops.expand_dims(
        tf.compat.v1.nn.ctc_loss(
            inputs=y_pred, labels=sparse_labels, sequence_length=input_length
        ),
        1,
    )


def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])

    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(
            max_num_labels_tns, current_input
        )

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = ops.reshape(
        ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape
    )
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)

    batch_array = ops.transpose(
        ops.reshape(
            ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(
        ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1])
    )

    vals_sparse = tf.compat.v1.gather_nd(labels, indices)

    return tf.SparseTensor(
        ops.cast(indices, dtype="int64"),
        vals_sparse,
        ops.cast(label_shape, dtype="int64"),
    )


class CTCLayer(layers.Layer):
    def __init__(self, name=None, trainable = True, dtype = None):
        super().__init__(name=name)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = ops.cast(ops.shape(y_true)[0], dtype="int64")
        input_length = ops.cast(ops.shape(y_pred)[1], dtype="int64")
        label_length = ops.cast(ops.shape(y_true)[1], dtype="int64")

        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred
    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "ctc_loss":self.loss_fn.ctc_batch_cost
            }
        )
        return config

    @classmethod
    def from_config(cls,config):
        return cls(**config)

img_width = 200
img_height = 50

model = load_model('model.keras', compile=False, custom_objects={
    "CTCLayer":CTCLayer})

input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
labels = layers.Input(name="label", shape=(None,), dtype="float32")
input_signature = [tf.TensorSpec([None, 200, 50, 1], tf.float32, name='image'), tf.TensorSpec([0, 0], tf.float32, name='label')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature)
onnx.save(onnx_model, "model.onnx")

```
