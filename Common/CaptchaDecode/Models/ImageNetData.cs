using Microsoft.ML.Data;

namespace Common.CaptchaDecode.Models
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        [ColumnName("image")]
        [VectorType(1, 200, 50, 1)]
        public float[] ImageData;

        [LoadColumn(1)]
        public string Label;
    }
}
