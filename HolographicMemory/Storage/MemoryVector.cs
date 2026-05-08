using System.Numerics;
using System.Numerics.Tensors;

namespace HolographicMemory.Storage;

/// <summary>
/// Base class for holographic memory vectors
/// </summary>
/// <typeparam name="TSelf"></typeparam>
public abstract class BaseMemoryVector<TSelf>
    : IEquatable<TSelf>
    where TSelf : BaseMemoryVector<TSelf>
{
    public string Name { get; }
    public ReadOnlyMemory<float> Vector { get; }
    public ReadOnlyMemory<Complex> FftVector { get; }
    public abstract MemoryVectorType Type { get; }

    internal HolographicStorage Parent { get; }
    
    private readonly int _hash;

    internal BaseMemoryVector(string name, ReadOnlyMemory<float> vector, ReadOnlyMemory<Complex> fftVector, HolographicStorage parent)
    {
        Name = name;
        Vector = vector;
        FftVector = fftVector;
        Parent = parent;

        _hash = HashCode.Combine(Name, parent.Id);
    }

    public bool Equals(TSelf? other)
    {
        if (other == null)
            return false;

        return Name == other.Name
            && Parent.Id.Equals(other.Parent.Id);
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
        TensorPrimitives.Multiply(arr, factor, arr);
        return Create(name, Parent, arr);
    }

    protected abstract TSelf Create(string name, HolographicStorage parent, float[] vec);
}

/// <summary>
/// A predicate linking two entities (e.g. eats)
/// </summary>
public class MemoryPredicate
    : BaseMemoryVector<MemoryPredicate>
{
    public override MemoryVectorType Type => MemoryVectorType.Predicate;

    internal MemoryPredicate(string name, ReadOnlyMemory<float> vector, ReadOnlyMemory<Complex> fftVector, HolographicStorage parent)
        : base(name, vector, fftVector, parent)
    {
    }

    protected override MemoryPredicate Create(string name, HolographicStorage parent, float[] vector)
    {
        return new MemoryPredicate(name, vector, parent.Fft(vector), parent);
    }
}

/// <summary>
/// An entity (e.g. "Alice" or "Pizza")
/// </summary>
public class MemoryEntity
    : BaseMemoryVector<MemoryEntity>
{
    public override MemoryVectorType Type => MemoryVectorType.Entity;

    internal MemoryEntity(string name, ReadOnlyMemory<float> vector, ReadOnlyMemory<Complex> fftVector, HolographicStorage parent)
        : base(name, vector, fftVector, parent)
    {
    }

    protected override MemoryEntity Create(string name, HolographicStorage parent, float[] vector)
    {
        return new MemoryEntity(name, vector, parent.Fft(vector), parent);
    }
}

/// <summary>
/// A property of an entity (e.g. "Female")
/// </summary>
public class MemoryProperty
    : BaseMemoryVector<MemoryProperty>
{
    public override MemoryVectorType Type => MemoryVectorType.Property;

    internal MemoryProperty(string name, ReadOnlyMemory<float> vector, ReadOnlyMemory<Complex> fftVector, HolographicStorage parent)
        : base(name, vector, fftVector, parent)
    {
    }

    protected override MemoryProperty Create(string name, HolographicStorage parent, float[] vector)
    {
        return new MemoryProperty(name, vector, parent.Fft(vector), parent);
    }
}

/// <summary>
/// The type of a memroy vector
/// </summary>
public enum MemoryVectorType
{
    /// <summary>
    /// An entity (e.g. Martin/Pizza)
    /// </summary>
    Entity,
    
    /// <summary>
    /// A predicate (e.g. Eats) relating 2 entities
    /// </summary>
    Predicate,
    
    /// <summary>
    /// A property (e.g. Male)
    /// </summary>
    Property
}