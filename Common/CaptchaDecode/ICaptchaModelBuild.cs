using Tensorflow.Keras.Engine;

namespace Common.CaptchaDecode
{
    public interface ICaptchaModelBuild
    {
        IModel BuildModel();
    }
}
