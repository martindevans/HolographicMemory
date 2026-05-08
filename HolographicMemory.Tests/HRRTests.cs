using FftFlat;
using HolographicMemory.Storage;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Tests;

[TestClass]
public sealed class HRRTests
{
    private const int Dimensions = 64;
    private const double ToleranceLoose = 0.05;   // for norm checks
    private const double SimilarityThreshold = 0.75; // cosine similarity after roundtrip

    private static FastFourierTransform CreateFft() => new(Dimensions);

    /// <summary>
    /// Creates a seeded random unit-length vector of floats.
    /// </summary>
    private static float[] RandomUnitVector(int seed)
    {
        var rng = new Random(seed);
        var v = new float[Dimensions];
        for (var i = 0; i < Dimensions; i++)
            v[i] = (float)(rng.NextDouble() * 2 - 1);
        HRR.Normalize(v, v);
        return v;
    }

    /// <summary>
    /// Cosine similarity between two already-normalised vectors (= dot product).
    /// </summary>
    private static double Similarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        return (double)Dot(a, b);
    }

    /// <summary>
    /// L2 norm of a float span.
    /// </summary>
    private static double L2Norm(ReadOnlySpan<float> v) => (double)Norm(v);

    // ------------------------------------------------------------------
    // Normalize tests
    // ------------------------------------------------------------------

    [TestMethod]
    public void Normalize_ProducesUnitVector()
    {
        var v = new[] { 3f, 0f, 4f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                        0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f };
        HRR.Normalize(v, v);
        Assert.AreEqual(1.0, L2Norm(v), ToleranceLoose);
    }

    [TestMethod]
    public void Normalize_RandomVector_ProducesUnitLength()
    {
        var v = RandomUnitVector(42);
        // Vector is already normalised; re-normalising must still give unit norm.
        HRR.Normalize(v, v);
        Assert.AreEqual(1.0, L2Norm(v), ToleranceLoose);
    }

    [TestMethod]
    public void Normalize_WritesToSeparateOutput()
    {
        var input = new float[Dimensions];
        input[0] = 5f;
        var output = new float[Dimensions];

        HRR.Normalize(input, output);

        Assert.AreEqual(1.0, L2Norm(output), ToleranceLoose);
        // Input should be unchanged
        Assert.AreEqual(5f, input[0], 1e-6f);
    }

    // ------------------------------------------------------------------
    // Bind tests
    // ------------------------------------------------------------------

    [TestMethod]
    public void Bind_ProducesUnitLengthVector()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(1);
        var b = RandomUnitVector(2);
        var result = new float[Dimensions];

        HRR.Bind(fft, (ReadOnlySpan<float>)a, (ReadOnlySpan<float>)b, result);

        Assert.AreEqual(1.0, L2Norm(result), ToleranceLoose);
    }

    [TestMethod]
    public void Bind_IsCommutative()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(10);
        var b = RandomUnitVector(20);
        var ab = new float[Dimensions];
        var ba = new float[Dimensions];

        HRR.Bind(fft, (ReadOnlySpan<float>)a, (ReadOnlySpan<float>)b, ab);
        HRR.Bind(fft, (ReadOnlySpan<float>)b, (ReadOnlySpan<float>)a, ba);

        // Circular convolution is commutative, so the two results should be identical.
        Assert.AreEqual(1.0, Similarity(ab, ba), ToleranceLoose);
    }

    [TestMethod]
    public void Bind_DifferentPairs_ProduceDifferentVectors()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(1);
        var b = RandomUnitVector(2);
        var c = RandomUnitVector(3);

        var ab = new float[Dimensions];
        var ac = new float[Dimensions];

        HRR.Bind(fft, (ReadOnlySpan<float>)a, (ReadOnlySpan<float>)b, ab);
        HRR.Bind(fft, (ReadOnlySpan<float>)a, (ReadOnlySpan<float>)c, ac);

        // ab and ac should be dissimilar
        Assert.IsLessThan(0.5, Math.Abs(Similarity(ab, ac)),
            $"Expected dissimilar vectors but got similarity {Similarity(ab, ac):F4}");
    }

    // ------------------------------------------------------------------
    // Unbind tests
    // ------------------------------------------------------------------

    [TestMethod]
    public void Unbind_AfterBind_RecoversBVector()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(100);
        var b = RandomUnitVector(200);
        var bound = new float[Dimensions];
        var recovered = new float[Dimensions];

        HRR.Bind(fft, a, b, bound);
        HRR.Unbind(fft, bound, a, recovered);

        var similarity = Similarity(recovered, b);
        Assert.IsGreaterThanOrEqualTo(SimilarityThreshold, similarity, $"Expected similarity >= {SimilarityThreshold} but got {similarity:F4}");
    }

    [TestMethod]
    public void Unbind_AfterBind_RecoversAVector()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(300);
        var b = RandomUnitVector(400);
        var bound = new float[Dimensions];
        var recovered = new float[Dimensions];

        HRR.Bind(fft, a, b, bound);
        HRR.Unbind(fft, bound, b, recovered);

        var similarity = Similarity(recovered, a);
        Assert.IsGreaterThanOrEqualTo(SimilarityThreshold, similarity,
            $"Expected similarity >= {SimilarityThreshold} but got {similarity:F4}");
    }

    [TestMethod]
    public void Unbind_WithWrongKey_DoesNotRecoverOriginal()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(500);
        var b = RandomUnitVector(600);
        var wrongKey = RandomUnitVector(700);
        var bound = new float[Dimensions];
        var recovered = new float[Dimensions];

        HRR.Bind(fft, a, b, bound);
        HRR.Unbind(fft, bound, wrongKey, recovered);

        var similarity = Math.Abs(Similarity(recovered, b));
        Assert.IsLessThan(SimilarityThreshold, similarity,
            $"Expected low similarity with wrong key but got {similarity:F4}");
    }

    [TestMethod]
    public void Unbind_ProducesUnitLengthVector()
    {
        var fft = CreateFft();
        var a = RandomUnitVector(1000);
        var b = RandomUnitVector(2000);
        var bound = new float[Dimensions];
        var recovered = new float[Dimensions];

        HRR.Bind(fft, a, b, bound);
        HRR.Unbind(fft, bound, a, recovered);

        Assert.AreEqual(1.0, L2Norm(recovered), ToleranceLoose);
    }
}
