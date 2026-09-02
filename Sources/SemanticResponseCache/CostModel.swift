import Dispatch

/// Monotonic nanoseconds. Injected so tests can drive the ledger with exact
/// numbers instead of timing real work.
public protocol TimeSource: Sendable {
    func now() -> UInt64
}

public struct MonotonicTimeSource: TimeSource {
    public init() {}
    public func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
}

/// Where the time went.
///
/// A server-side semantic cache saves dollars, and a saved dollar is a saved
/// dollar regardless of how long the lookup took. On device, with Foundation
/// Models at zero token cost, the thing being saved is *latency and battery*,
/// and a lookup that costs more than the generation it skips is a net loss —
/// even at a 100% hit rate. That makes "does this cache pay for itself" a
/// measurement, not an assumption, and this is the measurement.
public struct LatencyLedger: Sendable, Hashable {
    public private(set) var lookupNanos: UInt64 = 0
    public private(set) var lookups = 0
    public private(set) var generationNanos: UInt64 = 0
    public private(set) var generations = 0

    public init() {}

    mutating func recordLookup(nanos: UInt64) {
        lookupNanos = lookupNanos &+ nanos   // wraps only after ~584 years of continuous lookup
        lookups = Saturating.add(lookups, 1)
    }

    mutating func recordGeneration(nanos: UInt64) {
        generationNanos = generationNanos &+ nanos
        generations = Saturating.add(generations, 1)
    }

    public var meanLookupNanos: Double? {
        guard lookups > 0 else { return nil }
        return Double(lookupNanos) / Double(lookups)
    }

    public var meanGenerationNanos: Double? {
        guard generations > 0 else { return nil }
        return Double(generationNanos) / Double(generations)
    }

    /// The break-even verdict.
    ///
    /// Every request pays the lookup. A fraction `hitRate` of requests skip a
    /// generation. The cache pays iff
    ///
    ///     meanLookup < hitRate × meanGeneration
    ///
    /// `nil` until both sides have been measured at least once.
    public func breakEven(hitRate: Double?) -> BreakEven? {
        guard let lookup = meanLookupNanos, let generation = meanGenerationNanos,
              let hitRate, hitRate.isFinite else { return nil }
        let saved = hitRate * generation
        return BreakEven(meanLookupNanos: lookup,
                         meanGenerationNanos: generation,
                         hitRate: hitRate,
                         savedPerRequestNanos: saved - lookup)
    }
}

public struct BreakEven: Sendable, Hashable {
    public let meanLookupNanos: Double
    public let meanGenerationNanos: Double
    public let hitRate: Double
    /// Positive means the cache is a net win per request; negative means it is
    /// adding latency on average.
    public let savedPerRequestNanos: Double

    public var paysOff: Bool { savedPerRequestNanos > 0 }

    /// The hit rate at which the cache would break even with the current
    /// lookup and generation costs. Values above 1 mean it cannot pay off at
    /// any hit rate — the lookup is more expensive than the generation.
    public var breakEvenHitRate: Double? {
        guard meanGenerationNanos > 0 else { return nil }
        return meanLookupNanos / meanGenerationNanos
    }
}
