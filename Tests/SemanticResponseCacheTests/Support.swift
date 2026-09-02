import Foundation
@testable import SemanticResponseCache

/// Wraps an embedder and counts calls, so a test can assert the exact tier
/// answered *without* embedding.
final class CountingEmbedder: Embedder, @unchecked Sendable {
    private let inner: any Embedder
    private let lock = NSLock()
    private var _calls = 0

    init(_ inner: any Embedder) { self.inner = inner }

    var dimension: Int { inner.dimension }

    var calls: Int {
        lock.lock(); defer { lock.unlock() }
        return _calls
    }

    func embed(_ text: String) async throws -> Embedding {
        increment()
        return try await inner.embed(text)
    }

    private func increment() {
        lock.lock(); defer { lock.unlock() }
        _calls += 1
    }
}

/// Always throws — the "embedder is unavailable" failure mode.
struct FailingEmbedder: Embedder {
    var dimension: Int { 16 }
    func embed(_ text: String) async throws -> Embedding { throw EmbeddingError.degenerateVector }
}

/// Returns the same unit vector for every input. This is the *deliberately
/// broken* embedder: under it every prompt is maximally similar to every other,
/// so a cache with any threshold ≤ 1 serves the first answer for everything.
/// The shadow sampler is supposed to catch exactly this.
struct ConstantEmbedder: Embedder {
    let dimension = 16
    func embed(_ text: String) async throws -> Embedding {
        var raw = [Float](repeating: 0, count: dimension)
        raw[0] = 1
        // Force-free: `Embedding(normalizing:)` cannot fail for a unit basis vector.
        guard let embedding = Embedding(normalizing: raw) else { throw EmbeddingError.degenerateVector }
        return embedding
    }
}

/// Returns the wrong dimension — a second embedder feeding a cache built for
/// the first.
struct WrongDimensionEmbedder: Embedder {
    let dimension = 16
    func embed(_ text: String) async throws -> Embedding {
        guard let embedding = Embedding(normalizing: [Float](repeating: 1, count: 8)) else {
            throw EmbeddingError.degenerateVector
        }
        return embedding
    }
}

struct AlwaysSample: SampleDecider {
    mutating func shouldSample(rate: Double) -> Bool { rate > 0 }
}

struct NeverSample: SampleDecider {
    mutating func shouldSample(rate: Double) -> Bool { false }
}

/// A time source that advances by a scripted amount on each call.
final class ScriptedTime: TimeSource, @unchecked Sendable {
    private let lock = NSLock()
    private var current: UInt64 = 0
    private var steps: [UInt64]

    init(steps: [UInt64]) { self.steps = steps }

    func now() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        if !steps.isEmpty { current &+= steps.removeFirst() }
        return current
    }
}

/// A generator whose answer depends on the *exact* prompt — the model that
/// disagrees with itself across paraphrases. Every semantic hit against it is
/// a false hit by construction.
actor ExactPromptGenerator {
    private(set) var calls = 0
    func generate(_ prompt: String) async throws -> CachedResponse {
        calls += 1
        return CachedResponse(text: "answer for: \(prompt)")
    }
}

/// A generator that answers everything identically and slowly, for the
/// coalescing test.
actor SlowConstantGenerator {
    private(set) var calls = 0
    let delay: Duration
    init(delay: Duration) { self.delay = delay }
    func generate(_ prompt: String) async throws -> CachedResponse {
        calls += 1
        try await Task.sleep(for: delay)
        return CachedResponse(text: "constant")
    }
}

/// A policy that returns an id that is never in the cache.
struct BogusEviction: EvictionPolicy {
    let name = "Bogus"
    func victim(among entries: [CacheEntry]) -> CacheEntry.ID? { UInt64.max }
}

enum Fixtures {
    static func embedder() throws -> HashedTrigramEmbedder { try HashedTrigramEmbedder(dimension: 256) }

    static func cache(threshold: Float = 0.92, maxEntries: Int = 64, maxBytes: Int = 1 << 20,
                      embedder: any Embedder, eviction: any EvictionPolicy = CoverageAwareEviction(),
                      shadowRate: Double = 0, decider: any SampleDecider = NeverSample(),
                      timeSource: any TimeSource = MonotonicTimeSource()) throws -> SemanticCache {
        let policy = try CachePolicy(similarityThreshold: threshold, maxEntries: maxEntries, maxBytes: maxBytes)
        let shadow = try ShadowConfiguration(sampleRate: shadowRate)
        return SemanticCache(policy: policy, embedder: embedder, eviction: eviction,
                             shadow: shadow, decider: decider, timeSource: timeSource)
    }

    static func entry(id: UInt64, prompt: String, embedder: HashedTrigramEmbedder,
                      lastHit: Date, hits: Int = 0) throws -> CacheEntry {
        let embedding = try embedder.embedSynchronously(prompt)
        var entry = CacheEntry(id: id, normalizedPrompt: PromptNormalizer().normalize(prompt),
                               embedding: embedding, response: CachedResponse(text: "r"), createdAt: lastHit)
        var remaining = hits
        while remaining > 0 { entry.recordHit(at: lastHit); remaining -= 1 }
        return entry
    }
}
