using Microsoft.ML.Data;

namespace Common.CaptchaDecode.Models
{
    public class OnnxOutput
    {
        [ColumnName("ctc_loss")]
        [VectorType(1, 50, 21)]
        public float[] CtcLoss { get; set; }
    }
}
