/// A unit-length vector. The only kind the vector tier stores.
///
/// Normalising at construction means cosine similarity is a plain dot product
/// at lookup time — the hot path does one multiply-add per dimension and
/// nothing else. It also means a zero vector or a vector with a NaN component
/// can never enter the cache, because `init(normalizing:)` refuses them.
public struct Embedding: Sendable, Hashable {

    public let values: [Float]

    public var dimension: Int { values.count }

    /// Fails for an empty vector, a vector with any non-finite component, or a
    /// zero-norm vector. Those are the three inputs that would make a cosine
    /// similarity meaningless — and a meaningless similarity that clears a
    /// threshold is precisely how a semantic cache serves a wrong answer.
    public init?(normalizing raw: [Float]) {
        guard !raw.isEmpty else { return nil }
        var sumOfSquares: Float = 0
        for component in raw {
            guard component.isFinite else { return nil }
            sumOfSquares += component * component
        }
        guard sumOfSquares.isFinite, sumOfSquares > 0 else { return nil }
        let inverseNorm = 1 / sumOfSquares.squareRoot()
        guard inverseNorm.isFinite else { return nil }
        values = raw.map { $0 * inverseNorm }
    }

    /// Cosine similarity in `[-1, 1]`, or `nil` when the dimensions differ.
    ///
    /// A dimension mismatch is a programming error (two different embedders
    /// feeding one cache), but it is reported rather than trapped: the cache
    /// treats it as "not similar" and the metrics surface it. Rounding can push
    /// a dot product of two unit vectors fractionally outside `[-1, 1]`, so the
    /// result is clamped.
    public func cosine(_ other: Embedding) -> Float? {
        guard dimension == other.dimension, dimension > 0 else { return nil }
        var dot: Float = 0
        var index = 0
        while index < dimension {
            dot += values[index] * other.values[index]
            index += 1
        }
        guard dot.isFinite else { return nil }
        return min(1, max(-1, dot))
    }
}
