using HolographicMemory.Storage;

namespace HolographicMemory.Retrieval;

/// <summary>
/// A vector retrieved from vector DB search
/// </summary>
/// <typeparam name="TNumber"></typeparam>
public record RetrievalVector
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
    public ReadOnlyMemory<float> Value { get; }

    public RetrievalVector(Guid memoryId, string vectorId, MemoryVectorType type, ReadOnlyMemory<float> value)
    {
        MemoryId = memoryId;
        VectorId = vectorId;
        Type = type;
        Value = value;
    }
}