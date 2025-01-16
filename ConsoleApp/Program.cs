using Common.CaptchaDecode;

ICaptchaModelDecode captchaModelDecode = new CaptchaModelDecode(new List<string>() { "2", "3", "4", "5", "6", "7", "8", "b", "c", "d", "e", "f", "g", "m", "n", "p", "w", "x", "y" }, captchaLength: 5);
var result = captchaModelDecode.DecodeByFileName("test.png");
Console.WriteLine("result: " + result);
Console.ReadLine();