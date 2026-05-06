using System.Numerics;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory;

internal static class HRR
{
    public static void Bind<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fa = Borrow<Complex>.Get(a.Length);
        FFT(a, fa);

        using var fb = Borrow<Complex>.Get(a.Length);
        FFT(b, fb);

        Multiply(fa.Span, fb, fa);

        IFFT(fa, output);

        Normalize(output);
    }

    public static void Unbind<T>(ReadOnlySpan<T> memory, ReadOnlySpan<T> key, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fm = Borrow<Complex>.Get(memory.Length);
        FFT(memory, fm);

        using var fk = Borrow<Complex>.Get(memory.Length);
        FFT(key, fk);
        Conjugate(fk);

        Multiply(fm.Span, fk, fm);

        IFFT(fm, output);

        Normalize(output);
    }

    private static void FFT<T>(ReadOnlySpan<T> real, Span<Complex> output)
        where T : INumber<T>
    {
        // Ensure output is exactly the right length
        output = output[..real.Length];

        // Initialise with complex numbers
        for (var i = 0; i < real.Length; i++)
            output[i] = new Complex(double.CreateSaturating(real[i]), 0);

        FFT(output);
    }

    private static void FFT(Span<Complex> buffer)
    {
        new FftFlat.FastFourierTransform(buffer.Length).Forward(buffer);
    }

    private static void IFFT<T>(Span<Complex> freq, Span<T> output)
        where T : INumber<T>
    {
        IFFT(freq);

        for (var i = 0; i < freq.Length; i++)
            output[i] = T.CreateSaturating(freq[i].Real);
    }

    private static void IFFT(Span<Complex> buffer)
    {
        new FftFlat.FastFourierTransform(buffer.Length).Inverse(buffer);
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
        Divide(input, norm, output);
    }

    private static void Conjugate(Span<Complex> buffer)
    {
        for (var i = 0; i < buffer.Length; i++)
            buffer[i] = Complex.Conjugate(buffer[i]);
    }
}