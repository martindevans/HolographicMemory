using FftFlat;
using HolographicMemory.Exceptions;
using HolographicMemory.Extensions;
using System.Numerics;
using static HolographicMemory.Storage.HRR;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Storage;

public class HolographicStorage
{
    private readonly int _seed;
    private readonly FastFourierTransform _fft;
    
    /// <summary>
    /// The number of dimensions for vectors in this memory
    /// </summary>
    public int Dimensions { get; }

    /// <summary>
    /// Unique ID of this memory
    /// </summary>
    public Guid Id { get; }

    private readonly float[] _memoryVector;
    /// <summary>
    /// The raw (non normalised) memory vector
    /// </summary>
    public ReadOnlySpan<float> MemoryVector => _memoryVector;

    private readonly Complex[] _fftNormalizedMemory;
    private bool _requiresFft;
    /// <summary>
    /// The FFT of the normalised memory vector
    /// </summary>
    public ReadOnlySpan<Complex> FftNormalizedMemoryVector
    {
        get
        {
            if (_requiresFft)
            {
                // Normalize memory
                using var normalized = Borrow<float>.Get(_memoryVector.Length);
                Normalize(_memoryVector, normalized);
                
                // Calculate FFT of normalised memory
                FFT(_fft, normalized.Span, _fftNormalizedMemory);
                _requiresFft = false;
            }
            return _fftNormalizedMemory;
        }
    }

    public HolographicStorage(int dimensions, Guid id)
    {
        Dimensions = dimensions;
        Id = id;
        
        _memoryVector = new float[Dimensions];
        _fftNormalizedMemory = new Complex[Dimensions];
        _requiresFft = true;

        _seed = id.GetHashCode();
        _fft = new FastFourierTransform(Dimensions);
    }

    #region Create
    private ReadOnlyMemory<float> GenerateVector(string name, string type)
    {
        // Combine elements into seed
        var seed = HashCode.Combine(_seed, Dimensions);
        foreach (var ch in name)
            seed = HashCode.Combine(seed, char.ToLowerInvariant(ch));
        foreach (var ch in type)
            seed = HashCode.Combine(seed, char.ToLowerInvariant(ch));

        // Get seeded RNG
        var rng = new Random(seed);

        // Random [0, 1] numbers
        var arr = new float[Dimensions];
        for (var i = 0; i < arr.Length; i++)
            arr[i] = float.CreateChecked(rng.NextSingle());

        // Convert from [0, 1] to [-1, 1]
        Multiply(arr, float.CreateChecked(2f), arr);
        Subtract(arr, float.CreateChecked(1f), arr);

        // Normalise length
        Normalize(arr, arr);
        
        return arr;
    }

    /// <summary>
    /// Create a new entity vector (e.g. "Alice")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryEntity CreateEntity(string name)
    {
        var vec = GenerateVector(name, "ENTITY");
        var fftVec = Fft(vec.Span);
        return new MemoryEntity(name, vec, fftVec, this);
    }

    public MemoryEntity CreateEntity(string name, ReadOnlySpan<float> value)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, value.Length, nameof(value));
        return new MemoryEntity(name, value.ToArray(), Fft(value), this);
    }

    /// <summary>
    /// Create a new predicate vector (e.g. "Likes")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryPredicate CreatePredicate(string name)
    {
        var vec = GenerateVector(name, "PREDICATE");
        var fftVec = Fft(vec.Span);
        return new MemoryPredicate(name, vec, fftVec, this);
    }

    public MemoryPredicate CreatePredicate(string name, ReadOnlySpan<float> value)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, value.Length, nameof(value));
        return new MemoryPredicate(name, value.ToArray(), Fft(value), this);
    }

    /// <summary>
    /// Create a new property (e.g. "Female"). Can be used to add properties to things without having to do multi step is-a type relationships.
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryProperty CreateProperty(string name)
    {
        var vec = GenerateVector(name, "PROPERTY");
        var fftVec = Fft(vec.Span);
        return new MemoryProperty(name, vec, fftVec, this);
    }

    public MemoryProperty CreateProperty(string name, ReadOnlySpan<float> value)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, value.Length, nameof(value));
        return new MemoryProperty(name, value.ToArray(), Fft(value), this);
    }

    public MemoryProperty CreateProperty(string name, ReadOnlySpan<float> value, ReadOnlySpan<Complex> fftVal)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, value.Length, nameof(value));
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, fftVal.Length, nameof(fftVal));
        
        return new MemoryProperty(name, value.ToArray(), fftVal.ToArray(), this);
    }
    #endregion

    #region Store
    public void Store(Span<float> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        Add(_memoryVector, vector, _memoryVector);

        _requiresFft = true;
    }
    
    public void Store(MemoryEntity subject, MemoryPredicate predicate, MemoryEntity obj)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(predicate, this, nameof(predicate));
        VectorMismatchException.CheckAndThrow(obj, this, nameof(obj));

        using var bc = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, predicate.FftVector.Span, obj.FftVector.Span, bc);

        using var abc = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, subject.FftVector.Span, bc, abc);

        Store(abc);
    }

    public void Store(MemoryEntity subject, MemoryProperty property)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(property, this, nameof(property));

        using var ab = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, subject.FftVector.Span, property.FftVector.Span, ab);
        
        Store(ab);
    }
    #endregion

    #region Remove
    public void Remove(Span<float> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        Subtract(_memoryVector, vector, _memoryVector);
        
        _requiresFft = true;
    }

    public void Remove(MemoryEntity subject, MemoryPredicate predicate, MemoryEntity obj)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(predicate, this, nameof(predicate));
        VectorMismatchException.CheckAndThrow(obj, this, nameof(obj));

        using var bc = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, predicate.FftVector.Span, obj.FftVector.Span, bc);

        using var abc = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, subject.FftVector.Span, bc, abc);
        
        Remove(abc);
    }

    public void Remove(MemoryEntity subject, MemoryProperty property)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(property, this, nameof(property));

        using var ab = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, subject.FftVector.Span, property.FftVector.Span, ab);

        Remove(ab);
    }
    #endregion

    #region clear
    /// <summary>
    /// Clear everything from this memory
    /// </summary>
    public void Clear()
    {
        Array.Clear(_memoryVector);
        Array.Clear(_fftNormalizedMemory);
        _requiresFft = false;
    }

    /// <summary>
    /// Overwrite this memory with a raw vector value
    /// </summary>
    /// <param name="vector"></param>
    public void Set(ReadOnlySpan<float> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        vector.CopyTo(_memoryVector);
        _requiresFft = true;
    }
    #endregion

    #region query
    /// <summary>
    /// Given a Predicate and Object, extract matching Subjects.
    /// </summary>
    /// <param name="predicate">Predicate part of the query key.</param>
    /// <param name="obj">Object part of the query key.</param>
    /// <param name="output">Output span that receives the recovered subject-like vector.</param>
    public void QuerySubjects(MemoryPredicate predicate, MemoryEntity obj, Span<float> output)
    {
        VectorMismatchException.CheckAndThrow(predicate, this, nameof(predicate));
        VectorMismatchException.CheckAndThrow(obj, this, nameof(obj));

        // Calculate query vector
        using var key = Borrow<float>.Get(Dimensions);
        Bind<float>(_fft, predicate.FftVector.Span, obj.FftVector.Span, key);

        // Unbind from memory
        Unbind<float, float>(_fft, FftNormalizedMemoryVector, key, output);
    }

    /// <summary>
    /// Given a Subject and Predicate, extract matching Objects.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="predicate">Predicate part of the query key.</param>
    /// <param name="output">Output span that receives the recovered object-like vector.</param>
    public void QueryObjects(MemoryEntity subject, MemoryPredicate predicate, Span<float> output)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(predicate, this, nameof(predicate));

        using var x = Borrow<float>.Get(Dimensions);
        Unbind<float>(_fft, FftNormalizedMemoryVector, subject.FftVector.Span, x);

        Unbind(_fft, x, predicate.FftVector.Span, output);
    }

    /// <summary>
    /// Given a Subject and Object, extract matching Predicates.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="obj">Object part of the query key.</param>
    /// <param name="output">Output span that receives the recovered predicate-like vector.</param>
    public void QueryPredicates(MemoryEntity subject, MemoryEntity obj, Span<float> output)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));
        VectorMismatchException.CheckAndThrow(obj, this, nameof(obj));

        using var x = Borrow<float>.Get(Dimensions);
        Unbind<float>(_fft, FftNormalizedMemoryVector, subject.FftVector.Span, x);

        Unbind(_fft, x, obj.FftVector.Span, output);
    }

    /// <summary>
    /// Query a subject/property pair store with a subject key and recover matching properties.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="output">Output span that receives the recovered property-like vector.</param>
    public void QueryProperties(MemoryEntity subject, Span<float> output)
    {
        VectorMismatchException.CheckAndThrow(subject, this, nameof(subject));

        Unbind(_fft, FftNormalizedMemoryVector, subject.FftVector.Span, output);
    }

    /// <summary>
    /// Query a subject/property pair store with a property key and recover matching subjects.
    /// </summary>
    /// <param name="property">Property part of the query key.</param>
    /// <param name="output">Output span that receives the recovered subject-like vector.</param>
    public void QuerySubjects(MemoryProperty property, Span<float> output)
    {
        VectorMismatchException.CheckAndThrow(property, this, nameof(property));

        Unbind(_fft, FftNormalizedMemoryVector, property.FftVector.Span, output);
    }
    #endregion

    internal ReadOnlyMemory<Complex> Fft(ReadOnlySpan<float> vector)
    {
        var output = new Complex[vector.Length];
        FFT(_fft, vector, output.AsSpan());
        return output;
    }
}
