import Foundation

/// Chooses which entry leaves when the budget is exceeded.
///
/// The policy sees a snapshot of every entry and returns one victim. It is
/// called repeatedly until the cache is back under budget, so it must return a
/// member of `entries` (the cache verifies this and falls back to the oldest
/// entry if a policy misbehaves — an eviction policy bug must never become an
/// infinite loop).
public protocol EvictionPolicy: Sendable {
    var name: String { get }
    func victim(among entries: [CacheEntry]) -> CacheEntry.ID?
}

/// Least-recently-used. The reflex answer, included so the demo can measure
/// it against the alternative rather than assert the alternative is better.
///
/// Why it is the wrong default here: LRU assumes *recency predicts reuse*. For
/// URLs that is true. For semantic reuse it is not — the value of an entry is
/// how much of the prompt space it covers that nothing else does, and a
/// recently-hit entry sitting 0.97 away from another entry contributes almost
/// nothing, while an old entry that is the only one in its region is the
/// difference between a hit and a full generation.
public struct LRUEviction: EvictionPolicy {
    public let name = "LRU"
    public init() {}

    public func victim(among entries: [CacheEntry]) -> CacheEntry.ID? {
        entries.min { lhs, rhs in
            if lhs.lastHitAt != rhs.lastHitAt { return lhs.lastHitAt < rhs.lastHitAt }
            return lhs.id < rhs.id
        }?.id
    }
}

/// Evicts the most *redundant* entry: the one whose nearest neighbour in the
/// cache is closest to it, because that is the entry whose removal loses the
/// least coverage.
///
/// Ties (including the degenerate "nothing is near anything" case) break on
/// fewest hits, then oldest last hit, then lowest id, so the policy is total
/// and deterministic.
///
/// Cost is O(n²·d) per eviction. For the sizes an on-device cache holds — a
/// few hundred to a few thousand entries of a few hundred dimensions — that is
/// well under a millisecond and runs on the cache's own executor; eviction
/// happens on insert, which is already the slow path (a generation just
/// completed). Above ~10k entries an index would be warranted; that is a
/// documented limit, not a hidden one.
public struct CoverageAwareEviction: EvictionPolicy {
    public let name = "Coverage-aware"
    public init() {}

    public func victim(among entries: [CacheEntry]) -> CacheEntry.ID? {
        guard !entries.isEmpty else { return nil }
        if entries.count == 1 { return entries[0].id }

        var best: (id: UInt64, redundancy: Float, hits: Int, lastHit: Date)?
        var i = 0
        while i < entries.count {
            let candidate = entries[i]
            var nearest: Float = -1
            var j = 0
            while j < entries.count {
                if i != j, let similarity = candidate.embedding.cosine(entries[j].embedding) {
                    nearest = max(nearest, similarity)
                }
                j += 1
            }
            let current = (id: candidate.id, redundancy: nearest, hits: candidate.hitCount, lastHit: candidate.lastHitAt)
            if let existing = best {
                if Self.isWorse(current, than: existing) { best = current }
            } else {
                best = current
            }
            i += 1
        }
        return best?.id
    }

    /// "Worse" means "a better victim": more redundant, then fewer hits, then
    /// older, then lower id.
    private static func isWorse(_ a: (id: UInt64, redundancy: Float, hits: Int, lastHit: Date),
                                than b: (id: UInt64, redundancy: Float, hits: Int, lastHit: Date)) -> Bool {
        if a.redundancy != b.redundancy { return a.redundancy > b.redundancy }
        if a.hits != b.hits { return a.hits < b.hits }
        if a.lastHit != b.lastHit { return a.lastHit < b.lastHit }
        return a.id < b.id
    }
}
