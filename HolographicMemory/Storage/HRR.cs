using FftFlat;
using HolographicMemory.Extensions;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Storage;

internal static class HRR
{
    #region bind
    public static void Bind<T>(FastFourierTransform fft, ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fa = Borrow<Complex>.Get(a.Length);
        FFT(fft, a, fa);

        Bind(fft, fa, b, output);
    }

    public static void Bind<T>(FastFourierTransform fft, ReadOnlySpan<Complex> ffta, ReadOnlySpan<T> b, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fb = Borrow<Complex>.Get(b.Length);
        FFT(fft, b, fb);

        Bind(fft, ffta, fb.Span, output);
    }

    public static void Bind<T>(FastFourierTransform fft, ReadOnlySpan<Complex> ffta, ReadOnlySpan<Complex> fftb, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var temp = Borrow<Complex>.Get(ffta.Length);

        Multiply(ffta, fftb, temp);

        IFFT(fft, temp, output);

        Normalize(output);
    }
    #endregion

    #region unbind
    public static void Unbind<TA, TB, TC>(FastFourierTransform fft, ReadOnlySpan<TA> memory, ReadOnlySpan<TB> key, Span<TC> output)
        where TA : INumber<TA>, IRootFunctions<TA>
        where TB : INumber<TB>, IRootFunctions<TB>
        where TC : INumber<TC>, IRootFunctions<TC>
    {
        using var fm = Borrow<Complex>.Get(memory.Length);
        FFT(fft, memory, fm);

        using var fk = Borrow<Complex>.Get(memory.Length);
        FFT(fft, key, fk);

        Unbind(fft, fm, fk, output);
    }

    public static void Unbind<TB, TC>(FastFourierTransform fft, ReadOnlySpan<Complex> fftMemory, ReadOnlySpan<TB> key, Span<TC> output)
        where TB : INumber<TB>, IRootFunctions<TB>
        where TC : INumber<TC>, IRootFunctions<TC>
    {
        using var fk = Borrow<Complex>.Get(key.Length);
        FFT(fft, key, fk);

        Unbind(fft, fftMemory, fk, output);
    }

    public static void Unbind<T>(FastFourierTransform fft, ReadOnlySpan<T> memory, ReadOnlySpan<Complex> fftKey, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fm = Borrow<Complex>.Get(memory.Length);
        FFT(fft, memory, fm);

        Unbind(fft, fm, fftKey, output);
    }

    public static void Unbind<T>(FastFourierTransform fft, ReadOnlySpan<Complex> fftMemory, ReadOnlySpan<Complex> fftKey, Span<T> output)
        where T : INumber<T>, IRootFunctions<T>
    {
        using var fm = Borrow<Complex>.Get(fftMemory.Length);

        SemiConjugateMultiply(fftMemory, fftKey, fm);

        IFFT(fft, fm, output);

        Normalize(output);
    }
    #endregion

    internal static void FFT<T>(FastFourierTransform fft, ReadOnlySpan<T> real, Span<Complex> output)
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
        Multiply(input, T.One / Norm(input), output);
    }

    /// <summary>
    /// Compute `output = left * Conjugate(right)`
    /// </summary>
    /// <param name="left"></param>
    /// <param name="right"></param>
    /// <param name="output"></param>
    private static void SemiConjugateMultiply(ReadOnlySpan<Complex> left, ReadOnlySpan<Complex> right, Span<Complex> output)
    {
        if (left.Length != right.Length || left.Length != output.Length)
            throw new ArgumentException("Span lengths must match.");

        // Convert each complex to a vector128
        var lv = MemoryMarshal.Cast<Complex, Vector128<double>>(left);
        var rv = MemoryMarshal.Cast<Complex, Vector128<double>>(right);

        var leftShuffle = Vector128.Create(1L, 0L);

        for (var i = 0; i < left.Length; i++)
        {
            //output[i] = l * Complex.Conjugate(r);
            
            //output[i] = new Complex(
            //    (l.Real * r.Real) + (l.Imaginary * r.Imaginary),
            //    (l.Imaginary * r.Real) - (l.Real * r.Imaginary)
            //);
            
            // lr * rr + li * ri
            // li * rr - lr * ri

            // LR, LI, LI, LR
            var v_lr_li_li_lr = Vector256.Create(lv[i], Vector128.Shuffle(lv[i], leftShuffle));
            
            // RR, RI, RR, RI
            var v_rr_ri_rr_ri = Vector256.Create(rv[i], rv[i]);

            // LR * RR
            // LI * RI
            // LI * RR
            // LR * RI
            var mul = Vector256.Multiply(v_lr_li_li_lr, v_rr_ri_rr_ri);

            output[i] = new Complex(
                mul[0] + mul[1],
                mul[2] - mul[3]
            );
        }
    }
}