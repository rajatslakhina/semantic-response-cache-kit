/// The knobs a semantic cache exposes, validated at construction so that an
/// invalid configuration is a thrown error at startup rather than a cache that
/// quietly returns wrong answers.
public struct CachePolicy: Sendable, Hashable {

    /// Cosine similarity at or above which a stored response is returned for a
    /// different prompt. This is the number that makes a semantic cache
    /// different from every other cache: it is not a correctness invariant, it
    /// is a *tunable*, and every value of it trades hit rate against false-hit
    /// rate. Production guidance that consolidated through 2026 puts it around
    /// 0.92 with 0.92–0.97 as the working band — for server-side sentence
    /// embedders. The number for a different embedder is different, which is
    /// why `ShadowSampler` exists to measure it rather than assume it.
    public let similarityThreshold: Float

    /// Hard cap on stored entries.
    public let maxEntries: Int

    /// Hard cap on the summed `CacheEntry.byteCost`. On device this is a
    /// constraint, not a preference.
    public let maxBytes: Int

    /// The false-hit rate the deployment is willing to tolerate. 2% is the
    /// commonly quoted figure for non-regulated use, 0.5% for regulated.
    /// `ThresholdAdvisor` compares the *upper confidence bound* of the measured
    /// rate against this, not the point estimate.
    public let falseHitTolerance: Double

    public init(similarityThreshold: Float = 0.92,
                maxEntries: Int = 512,
                maxBytes: Int = 4 * 1024 * 1024,
                falseHitTolerance: Double = 0.02) throws {
        guard similarityThreshold.isFinite, similarityThreshold > 0, similarityThreshold <= 1 else {
            throw CacheError.invalidThreshold(similarityThreshold)
        }
        guard maxEntries >= 1 else { throw CacheError.invalidBudget("maxEntries must be >= 1, got \(maxEntries)") }
        guard maxBytes >= 1 else { throw CacheError.invalidBudget("maxBytes must be >= 1, got \(maxBytes)") }
        guard falseHitTolerance.isFinite, falseHitTolerance >= 0, falseHitTolerance <= 1 else {
            throw CacheError.invalidBudget("falseHitTolerance must be in [0, 1], got \(falseHitTolerance)")
        }
        self.similarityThreshold = similarityThreshold
        self.maxEntries = maxEntries
        self.maxBytes = maxBytes
        self.falseHitTolerance = falseHitTolerance
    }

    /// A copy with a different threshold; used by the advisor and the demo.
    public func withThreshold(_ threshold: Float) throws -> CachePolicy {
        try CachePolicy(similarityThreshold: threshold, maxEntries: maxEntries,
                        maxBytes: maxBytes, falseHitTolerance: falseHitTolerance)
    }
}

public enum CacheError: Error, Equatable, Sendable {
    case invalidThreshold(Float)
    case invalidBudget(String)
    case invalidSampleRate(Double)
}
