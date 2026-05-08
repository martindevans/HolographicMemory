using HolographicMemory.Storage;
using System.Numerics;

namespace HolographicMemory.Exceptions;

/// <summary>
/// Thrown if a vector created with one memory is used with another memory
/// </summary>
public class VectorMismatchException
    : ArgumentException
{
    private VectorMismatchException(string paramName)
        : base("Used vector intended for one memory with a different memory", paramName)
    {
    }

    internal static void CheckAndThrow<TV, TN>(TV vector, HolographicStorage<TN> memory, string paramName)
        where TV : BaseMemoryVector<TV, TN>
        where TN : struct, INumber<TN>, IRootFunctions<TN>
    {
        if (vector.Parent != memory)
            throw new VectorMismatchException(paramName);
    }
}