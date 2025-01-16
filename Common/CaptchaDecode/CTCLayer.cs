using Tensorflow.Common.Types;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Saving;
using Tensorflow.NumPy;
using Tensorflow.Operations;
using Tensorflow;

namespace Common.CaptchaDecode
{

    public class CTCLayer : Layer
    {
        public CTCLayer(LayerArgs args) : base(args)
        {

        }

        private Tensor ctc_batch_cost(Tensors y_true, Tensors? y_pred, Tensor? input_length, Tensor? label_length)
        {
            label_length = gen_math_ops.cast(gen_ops.squeeze(label_length, squeeze_dims: new int[] { -1 }), DstT: TF_DataType.TF_INT32);
            input_length = gen_math_ops.cast(gen_ops.squeeze(input_length, squeeze_dims: new int[] { -1 }), DstT: TF_DataType.TF_INT32);

            var sparse_labels = gen_math_ops.cast(ctc_label_dense_to_sparse(y_true, label_length), DstT: TF_DataType.TF_INT32);

            y_pred = gen_math_ops.log(gen_ops.transpose(y_pred, new Tensor(new NDArray(new int[] { 1, 0, 2 })) + KerasApi.keras.backend.epsilon()));

            return gen_ops.expand_dims(ctc_loss(inputs: y_pred, labels: sparse_labels, sequence_length: input_length), new Tensor(1));
        }

        private Tensor ctc_loss(Tensors inputs, Tensor labels, Tensor? sequence_length)
        {
            throw new NotImplementedException();
        }

        private Tensor ctc_label_dense_to_sparse(Tensors y_true, Tensor label_length)
        {
            throw new NotImplementedException();
        }

        protected override Tensors Call(Tensors inputs, Tensors state = null, bool? training = null, IOptionalArgs? optional_args = null)
        {

            var batch_len = gen_math_ops.cast(gen_ops.shape(inputs)[0], DstT: TF_DataType.TF_INT64);
            var input_length = gen_math_ops.cast(gen_ops.shape(state)[1], DstT: TF_DataType.TF_INT64);
            var label_length = gen_math_ops.cast(gen_ops.shape(inputs)[1], DstT: TF_DataType.TF_INT64);

            input_length = input_length * array_ops.ones(gen_ops.shape(batch_len, TF_DataType.TF_FLOAT), dtype: TF_DataType.TF_INT64);
            label_length = label_length * array_ops.ones(gen_ops.shape(batch_len, TF_DataType.TF_FLOAT), dtype: TF_DataType.TF_INT64);

            var loss = new Func<Tensor>(() =>
            {
                return ctc_batch_cost(inputs, state, input_length, label_length);
            });
            add_loss(loss);

            return state;

        }

        public override IKerasConfig get_config()
        {
            var config = base.get_config();
            return config;
        }

        public override Tensors Apply(Tensors inputs, Tensors states = null, bool? training = false, IOptionalArgs? optional_args = null)
        {
            return base.Apply(inputs, states, training, optional_args);
        }
    }
}
