using System.Numerics;
using System.Numerics.Tensors;

namespace HolographicMemory;

/// <summary>
/// Base class for holographic memory vectors
/// </summary>
/// <typeparam name="TSelf"></typeparam>
/// <typeparam name="TNumber"></typeparam>
public abstract class BaseMemoryVector<TSelf, TNumber>
    : IEquatable<TSelf>
    where TSelf : BaseMemoryVector<TSelf, TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    public string Name { get; init; }
    public ReadOnlyMemory<TNumber> Vector { get; init; }

    internal HolographicMemory<TNumber> Parent { get; }
    
    private readonly int _hash;

    internal BaseMemoryVector(string name, ReadOnlyMemory<TNumber> vector, HolographicMemory<TNumber> parent)
    {
        Name = name;
        Vector = vector;
        Parent = parent;

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
        TensorPrimitives.Multiply(arr, TNumber.CreateChecked(factor), arr);
        return Create(name, Parent, arr);
    }

    protected abstract TSelf Create(string name, HolographicMemory<TNumber> parent, TNumber[] vec);
}

/// <summary>
/// A predicate linking two entities (e.g. eats)
/// </summary>
/// <typeparam name="TNumber"></typeparam>
public class MemoryPredicate<TNumber>
    : BaseMemoryVector<MemoryPredicate<TNumber>, TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    internal MemoryPredicate(string name, ReadOnlyMemory<TNumber> vector, HolographicMemory<TNumber> parent)
        : base(name, vector, parent)
    {
    }

    protected override MemoryPredicate<TNumber> Create(string name, HolographicMemory<TNumber> parent, TNumber[] vec)
    {
        return new MemoryPredicate<TNumber>(name, vec, parent);
    }
}

/// <summary>
/// An entity (e.g. "Alice" or "Pizza")
/// </summary>
/// <typeparam name="TNumber"></typeparam>
public class MemoryEntity<TNumber>
    : BaseMemoryVector<MemoryEntity<TNumber>, TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    internal MemoryEntity(string name, ReadOnlyMemory<TNumber> vector, HolographicMemory<TNumber> parent)
        : base(name, vector, parent)
    {
    }

    protected override MemoryEntity<TNumber> Create(string name, HolographicMemory<TNumber> parent, TNumber[] vec)
    {
        return new MemoryEntity<TNumber>(name, vec, parent);
    }
}

/// <summary>
/// A property of an entity (e.g. "Female")
/// </summary>
/// <typeparam name="TNumber"></typeparam>
public class MemoryProperty<TNumber>
    : BaseMemoryVector<MemoryProperty<TNumber>, TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    internal MemoryProperty(string name, ReadOnlyMemory<TNumber> vector, HolographicMemory<TNumber> parent)
        : base(name, vector, parent)
    {
    }

    protected override MemoryProperty<TNumber> Create(string name, HolographicMemory<TNumber> parent, TNumber[] vec)
    {
        return new MemoryProperty<TNumber>(name, vec, parent);
    }
}