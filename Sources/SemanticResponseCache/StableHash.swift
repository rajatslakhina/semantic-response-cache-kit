/// A process-stable 64-bit hash (FNV-1a).
///
/// Swift's `Hasher` is deliberately seeded per process, so two launches of the
/// same app hash the same string to different values. That is the right choice
/// for in-memory dictionaries and the wrong one for a cache key that may be
/// persisted, logged, compared across devices, or used to coalesce in-flight
/// requests by identity. The exact tier therefore keys on this instead.
public enum StableHash {

    @usableFromInline static let offsetBasis: UInt64 = 0xcbf2_9ce4_8422_2325
    @usableFromInline static let prime: UInt64 = 0x0000_0100_0000_01b3

    /// FNV-1a over the UTF-8 bytes of `text`.
    @inlinable
    public static func fnv1a(_ text: String) -> UInt64 {
        var hash = offsetBasis
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }

    /// FNV-1a over arbitrary bytes.
    @inlinable
    public static func fnv1a<S: Sequence>(bytes: S) -> UInt64 where S.Element == UInt8 {
        var hash = offsetBasis
        for byte in bytes {
            hash ^= UInt64(byte)
            hash = hash &* prime
        }
        return hash
    }
}
