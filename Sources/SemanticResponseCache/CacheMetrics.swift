/// A Wilson score interval for a binomial proportion.
///
/// Used for the false-hit rate because the sample is small and the proportion
/// is near zero — exactly the regime where the naive `k/n ± 1.96·sqrt(p(1-p)/n)`
/// interval is wrong (it collapses to zero width at k = 0, which would let a
/// cache with 12 samples and no observed disagreement claim a 0% false-hit
/// rate with certainty). Wilson's upper bound at k = 0, n = 12 is ~24%, which
/// is the honest answer.
public struct WilsonInterval: Sendable, Hashable {
    public let estimate: Double
    public let lower: Double
    public let upper: Double
    public let samples: Int

    /// - Parameters:
    ///   - successes: count of the event (here: shadow disagreements).
    ///   - trials: total samples. Returns `nil` for zero trials — there is no
    ///     estimate, and pretending otherwise is the failure mode above.
    ///   - z: the normal quantile; 1.96 for 95%.
    public init?(successes: Int, trials: Int, z: Double = 1.96) {
        guard trials > 0, successes >= 0, successes <= trials, z.isFinite, z > 0 else { return nil }
        let n = Double(trials)
        let p = Double(successes) / n
        let z2 = z * z
        let denominator = 1 + z2 / n
        let centre = (p + z2 / (2 * n)) / denominator
        let halfWidth = (z * (p * (1 - p) / n + z2 / (4 * n * n)).squareRoot()) / denominator
        guard centre.isFinite, halfWidth.isFinite else { return nil }
        estimate = p
        lower = max(0, centre - halfWidth)
        upper = min(1, centre + halfWidth)
        samples = trials
    }
}

/// Counters the cache maintains. All increments saturate.
public struct CacheMetrics: Sendable, Hashable {
    public private(set) var requests = 0
    public private(set) var exactHits = 0
    public private(set) var semanticHits = 0
    public private(set) var misses = 0
    /// Hits whose provenance no longer matched the tool registry. Counted as
    /// misses as well; this is the *reason* for those misses.
    public private(set) var staleRejections = 0
    /// Entries removed to stay under budget.
    public private(set) var evictions = 0
    /// Concurrent identical misses that shared one in-flight generation.
    public private(set) var coalescedRequests = 0
    /// Prompts or responses that could not be stored (empty prompt, oversized
    /// response, embedder failure).
    public private(set) var uncacheable = 0
    public private(set) var shadowSamples = 0
    public private(set) var shadowDisagreements = 0
    public private(set) var shadowFailures = 0

    /// Best similarity seen on the vector tier for each miss that reached it —
    /// the histogram a threshold is tuned from. Bounded so it cannot grow
    /// without limit; oldest values fall off.
    public private(set) var nearMissSimilarities: [Float] = []
    public static let nearMissHistoryLimit = 256

    public init() {}

    public var hits: Int { Saturating.add(exactHits, semanticHits) }

    /// Hit rate over all requests, or `nil` before any request.
    public var hitRate: Double? {
        guard requests > 0 else { return nil }
        return Double(hits) / Double(requests)
    }

    /// Measured false-hit rate among *semantic* hits, with a 95% Wilson
    /// interval. `nil` until at least one shadow sample has completed.
    public var falseHitRate: WilsonInterval? {
        WilsonInterval(successes: shadowDisagreements, trials: shadowSamples)
    }

    mutating func recordRequest() { requests = Saturating.add(requests, 1) }
    mutating func recordExactHit() { exactHits = Saturating.add(exactHits, 1) }
    mutating func recordSemanticHit() { semanticHits = Saturating.add(semanticHits, 1) }
    mutating func recordMiss(nearest: Float?) {
        misses = Saturating.add(misses, 1)
        if let nearest, nearest.isFinite {
            nearMissSimilarities.append(nearest)
            if nearMissSimilarities.count > Self.nearMissHistoryLimit {
                nearMissSimilarities.removeFirst(nearMissSimilarities.count - Self.nearMissHistoryLimit)
            }
        }
    }
    mutating func recordStale() { staleRejections = Saturating.add(staleRejections, 1) }
    mutating func recordEviction() { evictions = Saturating.add(evictions, 1) }
    mutating func recordCoalesced() { coalescedRequests = Saturating.add(coalescedRequests, 1) }
    mutating func recordUncacheable() { uncacheable = Saturating.add(uncacheable, 1) }
    mutating func recordShadow(_ verdict: ShadowVerdict) {
        switch verdict {
        case .confirmed:
            shadowSamples = Saturating.add(shadowSamples, 1)
        case .falseHit:
            shadowSamples = Saturating.add(shadowSamples, 1)
            shadowDisagreements = Saturating.add(shadowDisagreements, 1)
        case .generatorFailed:
            shadowFailures = Saturating.add(shadowFailures, 1)
        }
    }
}

/// Turns the measured false-hit interval into a threshold recommendation.
public enum ThresholdAdvisor {

    public enum Recommendation: Sendable, Hashable {
        /// Fewer than `minimumSamples` shadow checks; nothing can be said yet.
        case insufficientEvidence(samples: Int, needed: Int)
        /// The upper confidence bound is inside tolerance.
        case hold
        /// The upper bound exceeds tolerance; raise the threshold to this.
        case raise(to: Float)
        /// The upper bound is comfortably inside tolerance; the threshold could
        /// come down to buy hit rate.
        case considerLowering(to: Float)
    }

    public static let minimumSamples = 30
    public static let step: Float = 0.01
    public static let ceiling: Float = 0.99

    public static func recommend(metrics: CacheMetrics, policy: CachePolicy) -> Recommendation {
        guard let interval = metrics.falseHitRate, interval.samples >= minimumSamples else {
            return .insufficientEvidence(samples: metrics.shadowSamples, needed: minimumSamples)
        }
        let current = policy.similarityThreshold
        if interval.upper > policy.falseHitTolerance {
            let raised = min(ceiling, current + step)
            return raised > current ? .raise(to: raised) : .hold
        }
        // Only suggest loosening when even the upper bound is well inside
        // tolerance — the instrument errs towards caution.
        if interval.upper < policy.falseHitTolerance / 4, current - step > 0 {
            return .considerLowering(to: current - step)
        }
        return .hold
    }
}
