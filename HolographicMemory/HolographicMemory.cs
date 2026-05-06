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

    private FastFourierTransform _fft;

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
    /// Create a new subject vector (e.g. "Alice")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemorySubject<TNumber> CreateSubject(string name)
    {
        return new MemorySubject<TNumber>(name, GenerateVector(name, "SUBJECT"));
    }

    public MemorySubject<TNumber> CreateSubject(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemorySubject<TNumber>(name, value.ToArray());
    }

    /// <summary>
    /// Create a new predicate vector (e.g. "Likes")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryPredicate<TNumber> CreatePredicate(string name)
    {
        return new MemoryPredicate<TNumber>(name, GenerateVector(name, "PREDICATE"));
    }

    public MemoryPredicate<TNumber> CreatePredicate(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryPredicate<TNumber>(name, value.ToArray());
    }

    /// <summary>
    /// Create a new object vector (e.g. "Pizza")
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryObject<TNumber> CreateObject(string name)
    {
        return new MemoryObject<TNumber>(name, GenerateVector(name, "OBJECT"));
    }

    public MemoryObject<TNumber> CreateObject(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryObject<TNumber>(name, value.ToArray());
    }

    /// <summary>
    /// Create a new property (e.g. "Female"). Can be used to add properties to things without having to do multi step is-a type relationships.
    /// </summary>
    /// <param name="name"></param>
    /// <returns></returns>
    public MemoryProperty<TNumber> CreateProperty(string name)
    {
        return new MemoryProperty<TNumber>(name, GenerateVector(name, "PROPERTY"));
    }

    public MemoryProperty<TNumber> CreateProperty(string name, ReadOnlySpan<TNumber> value)
    {
        return new MemoryProperty<TNumber>(name, value.ToArray());
    }
    #endregion

    #region Store
    public void Store(Span<TNumber> vector)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(Dimensions, vector.Length, nameof(vector));

        Add(_memoryVector, vector, _memoryVector);
        _requiresNormalization = true;
    }
    
    public void Store(MemorySubject<TNumber> subject, MemoryPredicate<TNumber> predicate, MemoryObject<TNumber> obj)
    {
        using var bc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, bc);

        using var abc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, bc, abc);

        Store(abc);
    }

    public void Store(MemorySubject<TNumber> subject, MemoryProperty<TNumber> property)
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

    public void Remove(MemorySubject<TNumber> subject, MemoryPredicate<TNumber> predicate, MemoryObject<TNumber> obj)
    {
        using var bc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, bc);

        using var abc = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, subject.Vector.Span, bc, abc);
        
        Remove(abc);
    }

    public void Remove(MemorySubject<TNumber> subject, MemoryProperty<TNumber> property)
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
    /// Get subject that complete a subject/predicate/object triple
    /// </summary>
    /// <param name="predicate"></param>
    /// <param name="obj"></param>
    /// <param name="output"></param>
    public void Query(MemoryPredicate<TNumber> predicate, MemoryObject<TNumber> obj, Span<TNumber> output)
    {
        // Calculate query vector
        using var key = Borrow<TNumber>.Get(Dimensions);
        Bind(_fft, predicate.Vector.Span, obj.Vector.Span, key);

        // Unbind from memory
        Unbind(_fft, NormalizedMemoryVector, key, output);
    }

    public void Query(MemorySubject<TNumber> subject, MemoryPredicate<TNumber> predicate, Span<TNumber> output)
    {
        using var x = Borrow<TNumber>.Get(Dimensions);
        Unbind(_fft, NormalizedMemoryVector, subject.Vector.Span, x);

        Unbind(_fft, x, predicate.Vector.Span, output);
    }
    #endregion
}

public abstract class BaseMemoryVector<TSelf, TNumber>
    : IEquatable<TSelf>
    where TSelf : BaseMemoryVector<TSelf, TNumber>
    where TNumber : INumber<TNumber>
{
    public string Name { get; init; }
    public ReadOnlyMemory<TNumber> Vector { get; init; }

    private readonly int _hash;

    internal BaseMemoryVector(string name, ReadOnlyMemory<TNumber> vector)
    {
        Name = name;
        Vector = vector;

        _hash = Name.GetHashCode();
        foreach (var item in vector.Span)
            _hash = HashCode.Combine(_hash, item);
    }

    public bool Equals(TSelf? other)
    {
        if (other == null)
            return false;
        
        return Name == other.Name
            && Vector.Span.SequenceEqual(other.Vector.Span);
    }

    public override bool Equals(object? obj)
    {
        if (obj is null)
            return false;
        if (ReferenceEquals(this, obj))
            return true;
        if (obj.GetType() != typeof(TSelf))
            return false;
        
        return Equals((TSelf)obj);
    }

    public override int GetHashCode()
    {
        return _hash;
    }

    /// <summary>
    /// Create a derived vector, by multiplying elements by a factor
    /// </summary>
    /// <param name="name"></param>
    /// <param name="factor"></param>
    /// <returns></returns>
    public TSelf Derive(string name, float factor)
    {
        var arr = Vector.ToArray();
        Multiply(arr, TNumber.CreateChecked(factor), arr);
        return Create(name, arr);
    }

    protected abstract TSelf Create(string name, TNumber[] vec);
}

public class MemorySubject<TNumber>
    : BaseMemoryVector<MemorySubject<TNumber>, TNumber> where TNumber : INumber<TNumber>
{
    internal MemorySubject(string name, ReadOnlyMemory<TNumber> vector)
        : base(name, vector)
    {
    }

    protected override MemorySubject<TNumber> Create(string name, TNumber[] vec)
    {
        return new MemorySubject<TNumber>(name, vec);
    }
}

public class MemoryPredicate<TNumber>
    : BaseMemoryVector<MemoryPredicate<TNumber>, TNumber> where TNumber : INumber<TNumber>
{
    internal MemoryPredicate(string name, ReadOnlyMemory<TNumber> vector)
        : base(name, vector)
    {
    }

    protected override MemoryPredicate<TNumber> Create(string name, TNumber[] vec)
    {
        return new MemoryPredicate<TNumber>(name, vec);
    }
}

public class MemoryObject<TNumber>
    : BaseMemoryVector<MemoryObject<TNumber>, TNumber> where TNumber : INumber<TNumber>
{
    internal MemoryObject(string name, ReadOnlyMemory<TNumber> vector)
        : base(name, vector)
    {
    }

    protected override MemoryObject<TNumber> Create(string name, TNumber[] vec)
    {
        return new MemoryObject<TNumber>(name, vec);
    }
}

public class MemoryProperty<TNumber>
    : BaseMemoryVector<MemoryProperty<TNumber>, TNumber> where TNumber : INumber<TNumber>
{
    internal MemoryProperty(string name, ReadOnlyMemory<TNumber> vector)
        : base(name, vector)
    {
    }

    protected override MemoryProperty<TNumber> Create(string name, TNumber[] vec)
    {
        return new MemoryProperty<TNumber>(name, vec);
    }
}