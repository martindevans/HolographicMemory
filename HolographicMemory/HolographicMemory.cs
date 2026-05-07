using System.Numerics;
using FftFlat;
using static System.Numerics.Tensors.TensorPrimitives;
using static HolographicMemory.HRR;

namespace HolographicMemory;

public class HolographicMemory<TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    public int Dimensions { get; }
    public int Seed { get; }

    private readonly TNumber[] _memoryVector;
    public ReadOnlySpan<TNumber> MemoryVector => _memoryVector;

    private readonly TNumber[] _normalizedMemory;
    private bool _requiresNormalization;
    public ReadOnlySpan<TNumber> NormalizedMemoryVector
    {
        get
        {
            if (_requiresNormalization)
            {
                Normalize(_memoryVector, _normalizedMemory);
                _requiresNormalization = false;
            }
            return _normalizedMemory;
        }
    }

    private readonly FastFourierTransform _fft;

    public HolographicMemory(int dimensions, int seed)
    {
        Dimensions = dimensions;
        Seed = seed;

        _memoryVector = new TNumber[Dimensions];
        _normalizedMemory = new TNumber[Dimensions];
        _requiresNormalization = true;

        _fft = new FastFourierTransform(Dimensions);
    }

    #region Create
    private ReadOnlyMemory<TNumber> GenerateVector(string name, string type)
    {
        // Combine elements into seed
        var seed = HashCode.Combine(Seed, Dimensions);
        foreach (var ch in name)
            seed = HashCode.Combine(seed, char.ToLowerInvariant(ch));
        foreach (var ch in type)
            seed = HashCode.Combine(seed, char.ToLowerInvariant(ch));

        // Get seeded RNG
        var rng = new Random(seed);

        // Random [0, 1] numbers
        var arr = new TNumber[Dimensions];
        for (var i = 0; i < arr.Length; i++)
            arr[i] = TNumber.CreateChecked(rng.NextSingle());

        // Convert from [0, 1] to [-1, 1]
        Multiply(arr, TNumber.CreateChecked(2f), arr);
        Subtract(arr, TNumber.CreateChecked(1f), arr);

        // Normalise length
        Normalize(arr, arr);
        
        return arr;
    }

    /// <summary>
    /// Create a new entity vector (e.g. "Alice")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryEntity<TNumber> CreateEntity(string name)
    {
        return new MemoryEntity<TNumber>(name, GenerateVector(name, "ENTITY"), this);
    }

    public MemoryEntity<TNumber> CreateEntity(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryEntity<TNumber>(name, value.ToArray(), this);
    }

    /// <summary>
    /// Create a new predicate vector (e.g. "Likes")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryPredicate<TNumber> CreatePredicate(string name)
    {
        return new MemoryPredicate<TNumber>(name, GenerateVector(name, "PREDICATE"), this);
    }

    public MemoryPredicate<TNumber> CreatePredicate(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryPredicate<TNumber>(name, value.ToArray(), this);
    }

    /// <summary>
    /// Create a new property (e.g. "Female"). Can be used to add properties to things without having to do multi step is-a type relationships.
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryProperty<TNumber> CreateProperty(string name)
    {
        return new MemoryProperty<TNumber>(name, GenerateVector(name, "PROPERTY"), this);
    }

    public MemoryProperty<TNumber> CreateProperty(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryProperty<TNumber>(name, value.ToArray(), this);
    }
    #endregion

    #region Store
    public void Store(Span<TNumber> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        Add(_memoryVector, vector, _memoryVector);
        _requiresNormalization = true;
    }
    
    public void Store(MemoryEntity<TNumber> subject, MemoryPredicate<TNumber> predicate, MemoryEntity<TNumber> obj)
    {
        using var bc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, bc);

        using var abc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, bc, abc);

        Store(abc);
    }

    public void Store(MemoryEntity<TNumber> subject, MemoryProperty<TNumber> property)
    {
        using var ab = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, property.Vector.Span, ab);
        
        Store(ab);
    }
    #endregion

    #region Remove
    public void Remove(Span<TNumber> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        Subtract(_memoryVector, vector, _memoryVector);
        _requiresNormalization = true;
    }

    public void Remove(MemoryEntity<TNumber> subject, MemoryPredicate<TNumber> predicate, MemoryEntity<TNumber> obj)
    {
        using var bc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, bc);

        using var abc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, bc, abc);
        
        Remove(abc);
    }

    public void Remove(MemoryEntity<TNumber> subject, MemoryProperty<TNumber> property)
    {
        using var ab = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, property.Vector.Span, ab);

        Remove(ab);
    }
    #endregion

    #region clear
    public void Clear()
    {
        Array.Clear(_memoryVector);
        Array.Clear(_normalizedMemory);
        _requiresNormalization = false;
    }
    #endregion

    #region query
    /// <summary>
    /// Given a Predicate and Object, extract matching Subjects.
    /// </summary>
    /// <param name="predicate">Predicate part of the query key.</param>
    /// <param name="obj">Object part of the query key.</param>
    /// <param name="output">Output span that receives the recovered subject-like vector.</param>
    public void QuerySubjects(MemoryPredicate<TNumber> predicate, MemoryEntity<TNumber> obj, Span<TNumber> output)
    {
        // Calculate query vector
        using var key = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, key);

        // Unbind from memory
        Unbind(_fft, NormalizedMemoryVector, key, output);
    }

    /// <summary>
    /// Given a Subject and Predicate, extract matching Objects.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="predicate">Predicate part of the query key.</param>
    /// <param name="output">Output span that receives the recovered object-like vector.</param>
    public void QueryObjects(MemoryEntity<TNumber> subject, MemoryPredicate<TNumber> predicate, Span<TNumber> output)
    {
        using var x = Borrow<TNumber>.Get(Dimensions);
        Unbind(_fft, NormalizedMemoryVector, subject.Vector.Span, x);

        Unbind(_fft, x, predicate.Vector.Span, output);
    }

    /// <summary>
    /// Given a Subject and Object, extract matching Predicates.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="obj">Object part of the query key.</param>
    /// <param name="output">Output span that receives the recovered predicate-like vector.</param>
    public void QueryPredicates(MemoryEntity<TNumber> subject, MemoryEntity<TNumber> obj, Span<TNumber> output)
    {
        using var x = Borrow<TNumber>.Get(Dimensions);
        Unbind(_fft, NormalizedMemoryVector, subject.Vector.Span, x);

        Unbind(_fft, x, obj.Vector.Span, output);
    }

    /// <summary>
    /// Query a subject/property pair store with a subject key and recover matching properties.
    /// </summary>
    /// <param name="subject">Subject part of the query key.</param>
    /// <param name="output">Output span that receives the recovered property-like vector.</param>
    public void QueryProperties(MemoryEntity<TNumber> subject, Span<TNumber> output)
    {
        Unbind(_fft, NormalizedMemoryVector, subject.Vector.Span, output);
    }

    /// <summary>
    /// Query a subject/property pair store with a property key and recover matching subjects.
    /// </summary>
    /// <param name="property">Property part of the query key.</param>
    /// <param name="output">Output span that receives the recovered subject-like vector.</param>
    public void QuerySubjects(MemoryProperty<TNumber> property, Span<TNumber> output)
    {
        Unbind(_fft, NormalizedMemoryVector, property.Vector.Span, output);
    }
    #endregion
}
