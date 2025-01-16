namespace Common.CaptchaDecode
{
    public interface ICaptchaModelDecode
    {
        List<string> Characters { get; set; }
        string Decode(byte[] fileBytes);
        string DecodeByFileName(string fileName = "test.png");
    }
}
