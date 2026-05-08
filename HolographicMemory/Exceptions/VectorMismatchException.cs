using HolographicMemory.Storage;

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

    internal static void CheckAndThrow<TV>(TV vector, HolographicStorage memory, string paramName)
        where TV : BaseMemoryVector<TV>
    {
        if (vector.Parent != memory)
            throw new VectorMismatchException(paramName);
    }
}