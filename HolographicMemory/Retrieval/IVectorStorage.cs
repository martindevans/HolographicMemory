using HolographicMemory.Storage;

namespace HolographicMemory.Retrieval;

public interface IVectorStorage
{
    /// <summary>
    /// Search vector storage for the nearest vectors to a supplied vector. In order of similarity descending.
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="type"></param>
    /// <param name="vector"></param>
    /// <param name="max"></param>
    /// <returns></returns>
    IEnumerable<(float Similarity, RetrievalVector Vector)> Search(Guid memory, MemoryVectorType type, ReadOnlyMemory<float> vector, int max);

    /// <summary>
    /// Search vector storage for the single nearest vector to a supplied vector.
    /// </summary>
    /// <param name="memory"></param>
    /// <param name="type"></param>
    /// <param name="vector"></param>
    /// <returns></returns>
    (float Similarity, RetrievalVector Vector)? Search(Guid memory, MemoryVectorType type, ReadOnlyMemory<float> vector);
}