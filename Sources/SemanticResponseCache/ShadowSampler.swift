/// Decides whether a given semantic hit is shadow-verified.
///
/// Injected so tests can make the decision deterministic; the default uses the
/// system RNG.
public protocol SampleDecider: Sendable {
    /// Returns `true` with probability `rate` (already validated to `[0, 1]`).
    mutating func shouldSample(rate: Double) -> Bool
}

public struct SystemRandomDecider: SampleDecider {
    public init() {}
    public mutating func shouldSample(rate: Double) -> Bool {
        if rate >= 1 { return true }
        if rate <= 0 { return false }
        return Double.random(in: 0 ..< 1) < rate
    }
}

/// SplitMix64 — a small, seedable generator so a test can say "sample the 1st,
/// 3rd and 7th hits" and mean it.
public struct SeededDecider: SampleDecider {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }

    public mutating func shouldSample(rate: Double) -> Bool {
        if rate >= 1 { return true }
        if rate <= 0 { return false }
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        // Top 53 bits → uniform double in [0, 1).
        let unit = Double(z >> 11) / Double(1 << 53)
        return unit < rate
    }
}

/// Decides whether a cached response and a freshly generated one "agree".
///
/// This is the ground truth the false-hit rate is measured against, and it is
/// a protocol because the right judge is domain-specific: exact text for
/// structured output, an embedding distance for prose, a human for regulated
/// use. The default is deliberately strict.
public protocol ResponseJudge: Sendable {
    func agrees(cached: CachedResponse, fresh: CachedResponse) -> Bool
}

/// Agrees iff the two responses normalize to the same text and carry the same
/// provenance. Strict on purpose: a judge that is too lenient makes the
/// false-hit rate look better than it is, which is the one direction this
/// instrument must never err in.
public struct NormalizedTextJudge: ResponseJudge {
    private let normalizer = PromptNormalizer()
    public init() {}
    public func agrees(cached: CachedResponse, fresh: CachedResponse) -> Bool {
        normalizer.normalize(cached.text) == normalizer.normalize(fresh.text)
            && cached.provenance == fresh.provenance
    }
}

/// How often semantic hits are re-run against the real generator.
///
/// This is the observability design for a cache with no server-side gateway.
/// On a server you compute the false-hit rate offline from logs. On device you
/// cannot ship the ground truth anywhere, so the cache has to *make* its own
/// ground truth: a fraction of hits also pay for the generation they skipped,
/// the two answers are compared locally, and only the verdict — not the
/// content — leaves the device.
public struct ShadowConfiguration: Sendable, Hashable {
    public let sampleRate: Double

    public init(sampleRate: Double) throws {
        guard sampleRate.isFinite, sampleRate >= 0, sampleRate <= 1 else {
            throw CacheError.invalidSampleRate(sampleRate)
        }
        self.sampleRate = sampleRate
    }

    public static let off: ShadowConfiguration = {
        // 0 is provably inside [0, 1]; the throwing init cannot fail for it.
        (try? ShadowConfiguration(sampleRate: 0)) ?? ShadowConfiguration(unchecked: 0)
    }()

    private init(unchecked rate: Double) { sampleRate = rate }
}

/// The outcome of a shadow check attached to a result.
public enum ShadowVerdict: Sendable, Hashable {
    /// The regenerated response agreed with the cached one.
    case confirmed
    /// It did not. The cache has already replaced the offending entry.
    case falseHit(fresh: CachedResponse)
    /// The regeneration itself failed; nothing was learned.
    case generatorFailed(String)
}
