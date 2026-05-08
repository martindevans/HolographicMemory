using System.Numerics;
using HolographicMemory.Storage;

namespace HolographicMemory.Retrieval;

/// <summary>
/// A vector retrieved from vector DB search
/// </summary>
/// <typeparam name="TNumber"></typeparam>
public record RetrievalVector<TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>
{
    /// <summary>
    /// The ID of the memory this vector belongs to
    /// </summary>
    public Guid MemoryId { get; }

    /// <summary>
    /// The name of this vector
    /// </summary>
    public string VectorId { get; }

    /// <summary>
    /// The type of this vector
    /// </summary>
    public MemoryVectorType Type { get; }

    /// <summary>
    /// The raw vector value
    /// </summary>
    public ReadOnlyMemory<TNumber> Value { get; }

    public RetrievalVector(Guid memoryId, string vectorId, MemoryVectorType type, ReadOnlyMemory<TNumber> value)
    {
        MemoryId = memoryId;
        VectorId = vectorId;
        Type = type;
        Value = value;
    }
}