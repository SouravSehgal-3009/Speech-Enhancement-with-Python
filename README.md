# Speech-Enhancement-with-Python
Speech enhancement is the task of taking a noisy speech input and producing an enhanced speech output. With the help of python libraries, Speech Enhancement has been done on a noisy audio file by three different ways: 

**1. Moving Average method (Time Domain Filtering):** The moving average filter is a simple Low Pass FIR (Finite Impulse Response) filter commonly used for regulating an array of sampled data/signal. It takes M samples of input at a time and takes the average of those to produce a single output point.

**2. Frequency Domain Filtering:** Frequency Domain Filters help in denoising the audio signalby removal of high or low frequency components.

**3. Spectral Subtraction :** The basic principle of Spectral Subtraction is that if we assume additive noise, then we can subtract the noise spectrum from the noisy speech spectrum and we will be left with the clean speech spectrum.
