using Tensorflow;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;

namespace Common.CaptchaDecode
{
    public class CaptchaModelBuild : ICaptchaModelBuild
    {
        private readonly List<string> characters;

        public CaptchaModelBuild(List<string> characters)
        {
            this.characters = characters;
        }

        public IModel BuildModel()
        {
            var inputImage = KerasApi.keras.layers.Input(shape: new Shape(Consts.ImageWidth, Consts.ImageHeight, 1), name: "image", dtype: TF_DataType.TF_FLOAT);
            var inputLabels = KerasApi.keras.layers.Input(shape: new Shape(new int[] { 0 }), name: "label");
            var conv2d = KerasApi.keras.layers.Conv2D(32, new Shape(3, 3), activation: "relu", kernel_initializer: "he_normal", padding: "same");
            var output = conv2d.Apply(inputImage);

            output = KerasApi.keras.layers.MaxPooling2D((2, 2)).Apply(output);

            var conv2d2 = KerasApi.keras.layers.Conv2D(64, (3, 3), activation: "relu", kernel_initializer: "he_normal", padding: "same");
            output = conv2d2.Apply(output);

            output = KerasApi.keras.layers.MaxPooling2D((2, 2)).Apply(output);

            var new_shape = new Shape(Consts.ImageWidth / 4, Consts.ImageHeight / 4 * 64);
            output = KerasApi.keras.layers.Reshape(target_shape: new_shape).Apply(output);
            output = KerasApi.keras.layers.Dense(64, activation: "relu").Apply(output);
            output = KerasApi.keras.layers.Dropout(0.2f).Apply(output);

            // RNNs
            output = KerasApi.keras.layers.Bidirectional(KerasApi.keras.layers.LSTM(128, return_sequences: true, dropout: 0.25f)).Apply(output);
            output = KerasApi.keras.layers.Bidirectional(KerasApi.keras.layers.LSTM(64, return_sequences: true, dropout: 0.25f)).Apply(output);

            output = KerasApi.keras.layers.Dense(characters.Count + 1, activation: "softmax").Apply(output);

            output = new CTCLayer(new LayerArgs() { Name = "ctc_loss" }).Apply(inputLabels, output);
            var model = KerasApi.keras.Model(inputs: new NDArray(new[] { inputImage, inputLabels }), outputs: output, name: "ocr");

            var opt = KerasApi.keras.optimizers.Adam();

            model.compile(optimizer: opt, loss: null);

            return model;
        }
    }
}
