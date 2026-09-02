/// Produces the embedding the vector tier compares.
///
/// This is the seam where a real app plugs in `NLEmbedding`, a Core AI model,
/// or a Foundation Models adapter. The cache neither knows nor cares which; it
/// only requires that the same text embeds to the same vector for the lifetime
/// of the cache — an embedder that changes under a populated cache silently
/// turns every stored vector into noise, which is why `dimension` is checked
/// on every comparison rather than trusted.
public protocol Embedder: Sendable {
    /// Number of components in every vector this embedder produces.
    var dimension: Int { get }
    /// Embeds `text`. Throws `EmbeddingError.emptyInput` for text with no
    /// usable content rather than returning a degenerate vector.
    func embed(_ text: String) async throws -> Embedding
}

public enum EmbeddingError: Error, Equatable, Sendable {
    /// The text normalized to nothing — no letters, no digits.
    case emptyInput
    /// The embedder returned a vector the cache cannot use (wrong dimension,
    /// non-finite component, or zero norm).
    case degenerateVector
    /// `dimension` outside the supported range.
    case invalidDimension(Int)
}

/// A dependency-free, deterministic, CPU-only embedder built on feature hashing
/// of words and character trigrams.
///
/// **What it is:** a real embedder, in the sense that paraphrases sharing
/// vocabulary land close together and unrelated prompts land far apart, so the
/// whole two-tier / threshold / eviction / shadow-sampling machinery above it
/// is exercised on genuine geometry rather than on stubs.
///
/// **What it is not:** a language model. It has no notion of synonymy — "refund"
/// and "money back" are orthogonal to it. A production deployment substitutes
/// an on-device sentence embedder; nothing else in the package changes. The
/// README says this plainly, and the tests that measure similarity use it
/// only to assert *relative* order (paraphrase > unrelated), never a specific
/// value a model would produce.
///
/// The output is deterministic across processes because it hashes with
/// `StableHash` rather than `Hasher`.
public struct HashedTrigramEmbedder: Embedder {

    public static let minimumDimension = 8
    public static let maximumDimension = 4096

    public let dimension: Int
    private let normalizer = PromptNormalizer()

    /// - Parameter dimension: in `[8, 4096]`. Smaller buckets collide more and
    ///   inflate similarity between unrelated prompts; larger ones cost lookup
    ///   time linearly. 256 is a reasonable default for a few thousand entries.
    public init(dimension: Int = 256) throws {
        guard dimension >= Self.minimumDimension, dimension <= Self.maximumDimension else {
            throw EmbeddingError.invalidDimension(dimension)
        }
        self.dimension = dimension
    }

    public func embed(_ text: String) async throws -> Embedding {
        try embedSynchronously(text)
    }

    /// The synchronous core; exposed so tests and the eviction policy's
    /// coverage computation can call it without an executor hop.
    public func embedSynchronously(_ text: String) throws -> Embedding {
        let tokens = normalizer.tokens(text)
        guard !tokens.isEmpty else { throw EmbeddingError.emptyInput }

        var accumulator = [Float](repeating: 0, count: dimension)
        // `dimension` is validated >= 8 in `init`, so `% dimension` cannot be a
        // division by zero and the resulting index is always in bounds.
        let buckets = UInt64(dimension)

        func accumulate(_ feature: String, weight: Float) {
            let hash = StableHash.fnv1a(feature)
            let index = Int(hash % buckets)
            // Use a bit that `% buckets` did not consume for the sign so that
            // the sign is independent of the bucket.
            let sign: Float = (hash >> 63) == 0 ? 1 : -1
            accumulator[index] += sign * weight
        }

        for token in tokens {
            accumulate("w:" + token, weight: 1.0)
            // Padded character trigrams: catch inflection ("refund"/"refunds")
            // and typos without a vocabulary.
            let padded = Array(("_" + token + "_").unicodeScalars)
            if padded.count >= 3 {
                var start = 0
                while start + 3 <= padded.count {
                    var trigram = "t:"
                    trigram.unicodeScalars.append(contentsOf: padded[start ..< start + 3])
                    accumulate(trigram, weight: 0.5)
                    start += 1
                }
            }
        }

        guard let embedding = Embedding(normalizing: accumulator) else {
            // Reachable only if every feature cancelled to exactly zero, which
            // feature hashing makes astronomically unlikely but not impossible.
            throw EmbeddingError.degenerateVector
        }
        return embedding
    }
}
