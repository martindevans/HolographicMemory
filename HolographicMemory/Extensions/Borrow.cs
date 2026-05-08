using System.Buffers;

namespace HolographicMemory.Extensions;

internal readonly record struct Borrow<T>
    : IDisposable
{
    private readonly T[] _array;
    private readonly int _length;
    private readonly ArrayPool<T> _pool;

    public Span<T> Span => Memory.Span;
    public Memory<T> Memory => _array.AsMemory(0, _length);

    public Borrow(T[] array, int length, ArrayPool<T> pool)
    {
        _array = array;
        _length = length;
        _pool = pool;
    }

    public void Dispose()
    {
        _pool.Return(_array);
    }

    public static Borrow<T> Get(int length)
    {
        var pool = ArrayPool<T>.Shared;
        var arr = pool.Rent(length);
        return new Borrow<T>(arr, length, pool);
    }

    public static implicit operator Span<T>(Borrow<T> self)
    {
        return self.Span;
    }
}