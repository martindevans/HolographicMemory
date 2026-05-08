using FftFlat;
using HolographicMemory.Extensions;
using System.Numerics;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Storage;

internal static class HRR
{
    public static void Bind<T>(FastFourierTransform fft, ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fa = Borrow<Complex>.Get(a.Length);
        FFT(fft, a, fa);

        using var fb = Borrow<Complex>.Get(a.Length);
        FFT(fft, b, fb);

        Multiply(fa.Span, fb, fa);

        IFFT(fft, fa, output);

        Normalize(output);
    }

    public static void Unbind<T>(FastFourierTransform fft, ReadOnlySpan<T> memory, ReadOnlySpan<T> key, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fm = Borrow<Complex>.Get(memory.Length);
        FFT(fft, memory, fm);

        using var fk = Borrow<Complex>.Get(memory.Length);
        FFT(fft, key, fk);
        Conjugate(fk);

        Multiply(fm.Span, fk, fm);

        IFFT(fft, fm, output);

        Normalize(output);
    }

    private static void FFT<T>(FastFourierTransform fft, ReadOnlySpan<T> real, Span<Complex> output)
        where T : INumber<T>
    {
        // Ensure output is exactly the right length
        output = output[..real.Length];

        // Initialise with complex numbers
        for (var i = 0; i < real.Length; i++)
            output[i] = new Complex(double.CreateSaturating(real[i]), 0);

        fft.Forward(output);
    }

    private static void IFFT<T>(FastFourierTransform fft, Span<Complex> freq, Span<T> output)
        where T : INumber<T>
    {
        fft.Inverse(freq);

        for (var i = 0; i < freq.Length; i++)
            output[i] = T.CreateSaturating(freq[i].Real);
    }

    private static void Normalize<T>(Span<T> v)
        where T : INumber<T>, IRootFunctions<T>
    {
        Normalize(v, v);
    }

    internal static void Normalize<T>(ReadOnlySpan<T> input, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        var norm = Norm(input);
        Multiply(input, T.One / Norm(input), output);
    }

    private static void Conjugate(Span<Complex> buffer)
    {
        for (var i = 0; i < buffer.Length; i++)
            buffer[i] = Complex.Conjugate(buffer[i]);
    }
}