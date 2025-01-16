using Microsoft.ML;
using Common.CaptchaDecode.Models;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Operations;
using System.Reflection;

namespace Common.CaptchaDecode
{

    public class CaptchaModelDecode : ICaptchaModelDecode
    {
        public List<string> Characters { get; set; }
        public int CaptchaLength { get; set; }

        public CaptchaModelDecode(List<string> characterList, int captchaLength = 6)
        {
            Characters = characterList;
            CaptchaLength = captchaLength;
        }

        /// <summary>
        /// DecodeByFileName
        /// </summary>
        /// <param name="fileName"></param>
        /// <returns></returns>
        public string DecodeByFileName(string fileName = "test.png")
        {
            return Decode(File.ReadAllBytes(fileName));
        }

        /// <summary>
        /// Decode
        /// </summary>
        /// <param name="fileBytes"></param>
        /// <returns></returns>
        public string Decode(byte[] fileBytes)
        {
            var inputData = TFImageToMLNetData(fileBytes);

            MLContext mlContext = new MLContext();

            var imageDataTarget = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>() { new ImageNetData() { ImageData = inputData } });

            var onnxModelPipeline = mlContext.Transforms.ApplyOnnxModel(
                outputColumnNames: new string[] { "ctc_loss" },
                inputColumnNames: new string[] { "image" },
                modelBytes: Assembly.GetAssembly(typeof(CaptchaModelDecode)).GetManifestResourceStream("Common.model.onnx"),
                shapeDictionary: new Dictionary<string, int[]>()
                                    {
                                       { "image", new[] { 1, Consts.ImageWidth, Consts.ImageHeight, 1} }
                                    },
                gpuDeviceId: null,
                fallbackToCpu: false);
            var dataTransform = onnxModelPipeline.Fit(imageDataTarget).Transform(imageDataTarget);

            var predictTransform = onnxModelPipeline.Fit(dataTransform);

            var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, OnnxOutput>(predictTransform);

            var prediction = onnxPredictionEngine.Predict(new ImageNetData() { ImageData = inputData });

            Shape shape = new Shape(1, Consts.ImageHeight, prediction.CtcLoss.Length / Consts.ImageHeight);
            NDArray array = new NDArray(prediction.CtcLoss, shape);

            var input_len = np.ones(array.shape.Slice(0, 1)) * array.shape.Slice(1, 1);

            var decodeText = ctc_decode(array, input_len);

            return decodeText;
        }

        private float[] TFImageToMLNetData(byte[] fileBytes)
        {
            Tensor image = new NDArray(fileBytes, Shape.Scalar, TF_DataType.TF_STRING);
            tensorflow.image_internal imageInternal = new tensorflow.image_internal();
            image = imageInternal.decode_image(image, channels: 1);
            image = imageInternal.convert_image_dtype(image, TF_DataType.TF_FLOAT);
            image = imageInternal.resize(image, new Shape(Consts.ImageHeight, Consts.ImageWidth));

            NDArray array1 = new NDArray(new[] { 1, 0, 2 });
            image = array_ops.transpose(image, array1);
            image = array_ops.expand_dims(image, axis: 0);

            var inputData = image.numpy().reshape(-1).numpy().ToArray<float>();

            return inputData;
        }

        private string ctc_decode(Tensor y_pred, Tensor? input_length, bool greedy = true, int beam_width = 100, int top_paths = 1)
        {
            var input_shape = y_pred.shape;
            var num_samples = input_shape[0];
            var num_steps = input_shape[1];

            NDArray array1 = new NDArray(new[] { 1, 0, 2 });
            y_pred = math_ops.log(array_ops.transpose(y_pred, array1) + KerasApi.keras.backend.epsilon());
            input_length = gen_ops.cast(input_length.numpy(), TF_DataType.TF_INT32).numpy();

            var decoded = gen_ctc_ops.ctc_greedy_decoder(inputs: y_pred.numpy(), sequence_length: input_length, merge_repeated: true);

            var sparseTensor = new SparseTensor(indices: decoded[0].numpy(), values: decoded[1].numpy(), new NDArray(new[] { num_samples, num_steps }));
            var decoded_dense = gen_sparse_ops.sparse_to_dense<long>(sparseTensor.indices, sparseTensor.dense_shape, sparseTensor.values, default_value: -1);
            var decoded_dense_numpy = decoded_dense.numpy();
            string result = "";
            for (int i = 0; i < CaptchaLength; i++)
            {
                result += Characters[decoded_dense_numpy[0][i] - 1];
            }
            return result;
        }
    }
}
