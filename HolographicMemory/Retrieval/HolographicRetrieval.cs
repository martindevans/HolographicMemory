using System.Numerics;
using HolographicMemory.Extensions;
using HolographicMemory.Storage;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Retrieval;

public class HolographicRetrieval<TNumber>
    where TNumber : struct, INumber<TNumber>, IRootFunctions<TNumber>, IFloatingPointIeee754<TNumber>
{
    /// <summary>
    /// How many vectors to probe when trying to find the next best
    /// </summary>
    public int SearchProbes { get; set; } = 5;

    /// <summary>
    /// Once a single result has been returned, searching for subsequent results will terminate once similarity
    /// is below this factor of the initial result (like Top-P).
    /// </summary>
    public float QualityFactor
    {
        get;
        set => field = Math.Clamp(value, 0, 1);
    } = 0.5f;


    private readonly IVectorStorage<TNumber> _storage;

    public HolographicRetrieval(IVectorStorage<TNumber> storage)
    {
        _storage = storage;
    }
    
    public IEnumerable<(float Similarity, RetrievalVector<TNumber> Vector)> Retrieve(Guid memory, MemoryVectorType type, ReadOnlyMemory<TNumber> query, int max)
    {
        if (max <= 0)
            yield break;
        
        // Copy search vector to some memory we can mutate
        using var residualVector = Borrow<TNumber>.Get(query.Length);
        using var normalisedResidual = Borrow<TNumber>.Get(query.Length);
        query.Span.CopyTo(residualVector.Span);

        // Keep track of everything yielded so far
        var yielded = new HashSet<string>();
        
        // Get the single best vector, using potentially faster single result search
        var maybeBest0 = _storage.Search(memory, type, residualVector.Memory);
        if (!maybeBest0.HasValue)
            yield break;
        var best0 = maybeBest0.Value;

        // Yield the best vector
        yielded.Add(best0.Vector.VectorId);
        yield return best0;

        // Remove contribution of best
        RemoveVectorInfluence(residualVector.Span, best0.Vector.Value.Span);
        
        // We yielded one item already
        max--;

        // Get more results
        while (max > 0)
        {
            max--;
            
            // Normalise the residual for querying
            var norm = Norm(residualVector.Span);
            if (norm < TNumber.CreateSaturating(0.01f) || norm <= TNumber.Epsilon)
                yield break;
            Divide(residualVector, norm, normalisedResidual);
            
            // Find the next best vectors. if they've all been yielded already exit now.
            var nextBest = _storage
                .Search(memory, type, normalisedResidual.Memory, SearchProbes)
                .Where(a => !yielded.Contains(a.Vector.VectorId))
                .OrderByDescending(a => a.Similarity)
                .FirstOrDefault();
            if (nextBest == default)
                break;

            // Stop yielding when results are junk
            if (nextBest.Similarity < QualityFactor * best0.Similarity)
                break;

            // Yield this item
            yielded.Add(nextBest.Vector.VectorId);
            yield return nextBest;

            // Remove contribution of item
            RemoveVectorInfluence(residualVector.Span, nextBest.Vector.Value.Span);
        }

        static void RemoveVectorInfluence(Span<TNumber> value, ReadOnlySpan<TNumber> subtracted)
        {
            // Remove influence of "subtracted" projected along "value". i.e.
            // value -= Dot(value, subtracted) * subtracted;

            var scale = Dot(value, subtracted) / Dot(subtracted, subtracted);
            FusedMultiplyAdd(subtracted, -scale, value, value);
        }
    }
}