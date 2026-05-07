using System.Numerics;

namespace HolographicMemory.Exceptions;

/// <summary>
/// Thrown if a vector created with one memory is used with another memory
/// </summary>
public class VectorMismatch
    : ArgumentException
{
    private VectorMismatch(string paramName)
        : base("Used vector intended for one memory with a different memory", paramName)
    {
    }

    public static void CheckAndThrow<TV, TN>(TV vector, HolographicMemory<TN> memory, string paramName)
        where TV : BaseMemoryVector<TV, TN>
        where TN : struct, INumber<TN>, IRootFunctions<TN>
    {
        if (vector.Parent != memory)
            throw new VectorMismatch(paramName);
    }
}