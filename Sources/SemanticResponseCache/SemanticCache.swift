import Foundation

/// What `SemanticCache.respond` returns.
public struct CacheResult: Sendable, Hashable {
    public let response: CachedResponse
    public let source: Source
    /// Present only when this request was a semantic hit that the shadow
    /// sampler chose to verify.
    public let shadow: ShadowVerdict?

    public enum Source: Sendable, Hashable {
        /// Tier 1: the normalized prompt had been seen verbatim.
        case exactHit
        /// Tier 2: a different prompt cleared the similarity threshold.
        case semanticHit(similarity: Float, matchedPrompt: String)
        /// Neither tier answered; the generator ran.
        case generated(MissReason)
    }

    public var isHit: Bool {
        switch source {
        case .exactHit, .semanticHit: return true
        case .generated: return false
        }
    }
}

public enum MissReason: Sendable, Hashable {
    /// The cache had no entries to compare against.
    case cold
    /// The nearest stored prompt was this similar — not enough.
    case belowThreshold(nearest: Float)
    /// A stored entry matched but its provenance no longer matched the tool
    /// registry; it was discarded and the generator re-ran.
    case stale
    /// The prompt normalized to nothing; it was answered but not stored.
    case uncacheablePrompt
    /// The embedder threw; the prompt was answered but not stored.
    case embeddingFailed
    /// An identical prompt was already being generated; this request shared
    /// that generation instead of starting another.
    case coalesced
}

/// A read-only view of the cache's state for UIs and tests.
public struct CacheSnapshot: Sendable {
    public let policy: CachePolicy
    public let entries: [CacheEntry]
    public let totalBytes: Int
    public let metrics: CacheMetrics
    public let ledger: LatencyLedger
    public let toolVersions: [String: String]
    public let evictionPolicyName: String
}

/// An on-device semantic response cache.
///
/// ## The two tiers
/// 1. **Exact tier** — a dictionary keyed on `StableHash(normalize(prompt))`.
///    Costs a string walk and a hash. Answers near-verbatim repeats without
///    touching the embedder.
/// 2. **Vector tier** — a linear scan of every stored unit vector against the
///    prompt's embedding. Costs one embedding plus `n × d` multiply-adds. Only
///    reached when tier 1 misses.
///
/// A linear scan rather than an ANN index is a deliberate call: for the sizes
/// an on-device, per-user cache holds (hundreds to low thousands of entries)
/// it is faster than an index's constant factors, has no build step, and is
/// exactly correct — an approximate index adds a *second* probabilistic layer
/// on top of the threshold, and one is enough.
///
/// ## Reentrancy
/// This is an actor, and `respond` suspends twice: at the embedder and at the
/// generator. Other callers run between those points. Every suspension is
/// followed by re-reading the state it depends on — the exact tier is
/// re-checked after embedding, and the exact tier is consulted again before
/// storing so a concurrent insert of the same key updates in place rather than
/// producing a duplicate. Identical misses in flight at the same time share one
/// generation (`inFlight`). None of this is visible to callers; all of it is
/// what makes the budget invariant hold under concurrency, and there is a test
/// with real concurrent writers to say so.
///
/// ## Per-user isolation
/// One instance is one user. There is no cross-user warming, by design: a
/// semantic hit from another user's prompt is a privacy leak with a confidence
/// score. The consequence — cold start is permanent, and the hit-rate curve
/// never looks like the server benchmarks — is disclosed rather than hidden.
public actor SemanticCache {

    public let policy: CachePolicy
    private let embedder: any Embedder
    private let eviction: any EvictionPolicy
    private let judge: any ResponseJudge
    private let shadow: ShadowConfiguration
    private var decider: any SampleDecider
    private let time: any TimeSource
    private let dateProvider: @Sendable () -> Date
    private let normalizer = PromptNormalizer()

    private var entries: [CacheEntry.ID: CacheEntry] = [:]
    private var exactIndex: [UInt64: CacheEntry.ID] = [:]
    private var totalBytes = 0
    private var nextID: UInt64 = 1
    private var toolVersions: [String: String] = [:]
    private var inFlight: [UInt64: Task<CachedResponse, any Error>] = [:]
    private var metrics = CacheMetrics()
    private var ledger = LatencyLedger()

    public init(policy: CachePolicy,
                embedder: any Embedder,
                eviction: any EvictionPolicy = CoverageAwareEviction(),
                shadow: ShadowConfiguration = .off,
                judge: any ResponseJudge = NormalizedTextJudge(),
                decider: any SampleDecider = SystemRandomDecider(),
                timeSource: any TimeSource = MonotonicTimeSource(),
                dateProvider: @escaping @Sendable () -> Date = { Date() }) {
        self.policy = policy
        self.embedder = embedder
        self.eviction = eviction
        self.shadow = shadow
        self.judge = judge
        self.decider = decider
        self.time = timeSource
        self.dateProvider = dateProvider
    }

    // MARK: - Public API

    /// Answers `prompt` from the cache if it can, otherwise from `generate`.
    ///
    /// `generate` is the real model call. It is invoked at most once per
    /// distinct normalized prompt in flight, plus once more per shadow sample.
    public func respond(to prompt: String,
                        generate: @escaping @Sendable (String) async throws -> CachedResponse) async throws -> CacheResult {
        metrics.recordRequest()
        let started = time.now()

        let normalized = normalizer.normalize(prompt)
        guard !normalized.isEmpty else {
            metrics.recordUncacheable()
            ledger.recordLookup(nanos: elapsed(since: started))
            let response = try await timedGeneration(prompt, generate)
            return CacheResult(response: response, source: .generated(.uncacheablePrompt), shadow: nil)
        }
        let key = StableHash.fnv1a(normalized)

        // Tier 1.
        var sawStale = false
        switch exactLookup(key: key) {
        case .hit(let entry):
            ledger.recordLookup(nanos: elapsed(since: started))
            return CacheResult(response: entry.response, source: .exactHit, shadow: nil)
        case .stale:
            sawStale = true
        case .none:
            break
        }

        // Tier 2 — embed. This suspends; state may change underneath us.
        let embedding: Embedding?
        do {
            let produced = try await embedder.embed(prompt)
            embedding = produced.dimension == embedder.dimension ? produced : nil
        } catch {
            embedding = nil
        }

        // Re-check tier 1: a concurrent caller may have stored this exact key
        // while we were embedding.
        switch exactLookup(key: key) {
        case .hit(let entry):
            ledger.recordLookup(nanos: elapsed(since: started))
            return CacheResult(response: entry.response, source: .exactHit, shadow: nil)
        case .stale:
            sawStale = true
        case .none:
            break
        }

        var nearest: Float?
        if let embedding {
            let scan = vectorLookup(embedding)
            nearest = scan.nearest
            if scan.rejectedStale { sawStale = true }
            if let hit = scan.hit {
                ledger.recordLookup(nanos: elapsed(since: started))
                let verdict = await shadowVerify(hit: hit, prompt: prompt, normalized: normalized,
                                                 key: key, embedding: embedding, generate: generate)
                return CacheResult(response: hit.entry.response,
                                   source: .semanticHit(similarity: hit.similarity,
                                                        matchedPrompt: hit.entry.normalizedPrompt),
                                   shadow: verdict)
            }
        }

        // Miss.
        ledger.recordLookup(nanos: elapsed(since: started))
        metrics.recordMiss(nearest: nearest)

        let reason: MissReason
        if embedding == nil {
            reason = .embeddingFailed
        } else if sawStale {
            reason = .stale
        } else if let nearest {
            reason = .belowThreshold(nearest: nearest)
        } else {
            reason = .cold
        }

        // Coalesce with an identical in-flight generation.
        if let pending = inFlight[key] {
            metrics.recordCoalesced()
            let response = try await pending.value
            return CacheResult(response: response, source: .generated(.coalesced), shadow: nil)
        }

        let task = Task.detached { try await generate(prompt) }
        inFlight[key] = task
        let generationStarted = time.now()
        let response: CachedResponse
        do {
            response = try await task.value
        } catch {
            inFlight.removeValue(forKey: key)
            throw error
        }
        inFlight.removeValue(forKey: key)
        ledger.recordGeneration(nanos: elapsed(since: generationStarted))

        if let embedding {
            store(normalized: normalized, key: key, embedding: embedding, response: response)
        } else {
            metrics.recordUncacheable()
        }
        return CacheResult(response: response, source: .generated(reason), shadow: nil)
    }

    /// Declares the current data version of a tool. Every stored entry whose
    /// provenance names this tool at a *different* version becomes stale and
    /// is discarded the next time it would have been served.
    public func registerTool(_ tool: String, version: String) {
        toolVersions[tool] = version
    }

    /// Eagerly removes every entry grounded by `tool`, at any version.
    @discardableResult
    public func invalidate(tool: String) -> Int {
        let victims = entries.values.filter { entry in
            entry.response.provenance.contains { $0.tool == tool }
        }
        for victim in victims { remove(victim.id) }
        return victims.count
    }

    /// Eagerly removes every entry that `registerTool` has made stale.
    @discardableResult
    public func purgeStale() -> Int {
        let victims = entries.values.filter { !isFresh($0) }
        for victim in victims {
            remove(victim.id)
            metrics.recordStale()
        }
        return victims.count
    }

    public func removeAll() {
        entries.removeAll()
        exactIndex.removeAll()
        totalBytes = 0
    }

    public func snapshot() -> CacheSnapshot {
        CacheSnapshot(policy: policy,
                      entries: entries.values.sorted { $0.id < $1.id },
                      totalBytes: totalBytes,
                      metrics: metrics,
                      ledger: ledger,
                      toolVersions: toolVersions,
                      evictionPolicyName: eviction.name)
    }

    public var count: Int { entries.count }

    public func recommendation() -> ThresholdAdvisor.Recommendation {
        ThresholdAdvisor.recommend(metrics: metrics, policy: policy)
    }

    public func breakEven() -> BreakEven? {
        ledger.breakEven(hitRate: metrics.hitRate)
    }

    // MARK: - Tier 1

    private enum ExactLookup {
        case hit(CacheEntry)
        case stale
        case none
    }

    private func exactLookup(key: UInt64) -> ExactLookup {
        guard let id = exactIndex[key], var entry = entries[id] else { return .none }
        guard isFresh(entry) else {
            remove(id)
            metrics.recordStale()
            return .stale
        }
        entry.recordHit(at: dateProvider())
        entries[id] = entry
        metrics.recordExactHit()
        return .hit(entry)
    }

    // MARK: - Tier 2

    private struct SemanticHit {
        let entry: CacheEntry
        let similarity: Float
    }

    private struct VectorScan {
        var hit: SemanticHit?
        var nearest: Float?
        var rejectedStale = false
    }

    private func vectorLookup(_ query: Embedding) -> VectorScan {
        var scan = VectorScan()
        // Candidates at or above threshold, best first, so a stale best can be
        // discarded and the next-best considered in the same pass.
        var candidates: [(id: CacheEntry.ID, similarity: Float)] = []
        for entry in entries.values {
            guard let similarity = entry.embedding.cosine(query) else { continue }
            if let current = scan.nearest {
                if similarity > current { scan.nearest = similarity }
            } else {
                scan.nearest = similarity
            }
            if similarity >= policy.similarityThreshold {
                candidates.append((entry.id, similarity))
            }
        }
        candidates.sort { lhs, rhs in
            if lhs.similarity != rhs.similarity { return lhs.similarity > rhs.similarity }
            return lhs.id < rhs.id
        }
        for candidate in candidates {
            guard var entry = entries[candidate.id] else { continue }
            guard isFresh(entry) else {
                remove(entry.id)
                metrics.recordStale()
                scan.rejectedStale = true
                continue
            }
            entry.recordHit(at: dateProvider())
            entries[entry.id] = entry
            metrics.recordSemanticHit()
            scan.hit = SemanticHit(entry: entry, similarity: candidate.similarity)
            return scan
        }
        return scan
    }

    // MARK: - Shadow verification

    private func shadowVerify(hit: SemanticHit, prompt: String, normalized: String, key: UInt64,
                              embedding: Embedding,
                              generate: @escaping @Sendable (String) async throws -> CachedResponse) async -> ShadowVerdict? {
        guard decider.shouldSample(rate: shadow.sampleRate) else { return nil }
        let verdict: ShadowVerdict
        do {
            let fresh = try await timedGeneration(prompt, generate)
            if judge.agrees(cached: hit.entry.response, fresh: fresh) {
                verdict = .confirmed
            } else {
                verdict = .falseHit(fresh: fresh)
                // Self-heal: the entry that answered wrongly leaves, and the
                // correct answer for *this* prompt is stored under its own key.
                // The entry may already be gone (evicted during the await); both
                // removals are no-ops in that case.
                remove(hit.entry.id)
                store(normalized: normalized, key: key, embedding: embedding, response: fresh)
            }
        } catch {
            verdict = .generatorFailed(String(describing: error))
        }
        metrics.recordShadow(verdict)
        return verdict
    }

    // MARK: - Storage

    private func store(normalized: String, key: UInt64, embedding: Embedding, response: CachedResponse) {
        // A concurrent caller may have stored this key while we generated.
        // Replace rather than duplicate: the exact index must stay one-to-one.
        if let existing = exactIndex[key] { remove(existing) }

        let entry = CacheEntry(id: nextID, normalizedPrompt: normalized, embedding: embedding,
                               response: response, createdAt: dateProvider())
        nextID = nextID &+ 1
        guard entry.byteCost <= policy.maxBytes else {
            metrics.recordUncacheable()
            return
        }
        entries[entry.id] = entry
        exactIndex[key] = entry.id
        totalBytes = Saturating.add(totalBytes, entry.byteCost)
        enforceBudget()
    }

    private func enforceBudget() {
        while entries.count > policy.maxEntries || totalBytes > policy.maxBytes {
            guard !entries.isEmpty else { return } // unreachable: over-budget implies non-empty
            let snapshot = Array(entries.values)
            var victim = eviction.victim(among: snapshot)
            if let chosen = victim, entries[chosen] == nil { victim = nil }
            // A policy that returns nothing, or something not in the cache,
            // must not stall eviction: fall back to the oldest entry.
            let id = victim ?? snapshot.min { $0.id < $1.id }.map(\.id)
            guard let id else { return } // unreachable: snapshot is non-empty
            remove(id)
            metrics.recordEviction()
        }
    }

    private func remove(_ id: CacheEntry.ID) {
        guard let entry = entries.removeValue(forKey: id) else { return }
        let key = StableHash.fnv1a(entry.normalizedPrompt)
        if exactIndex[key] == id { exactIndex.removeValue(forKey: key) }
        totalBytes = max(0, totalBytes - entry.byteCost)
    }

    private func isFresh(_ entry: CacheEntry) -> Bool {
        for edge in entry.response.provenance {
            if let current = toolVersions[edge.tool], current != edge.version { return false }
        }
        return true
    }

    // MARK: - Timing

    private func elapsed(since start: UInt64) -> UInt64 {
        let now = time.now()
        return now >= start ? now - start : 0
    }

    private func timedGeneration(_ prompt: String,
                                 _ generate: @escaping @Sendable (String) async throws -> CachedResponse) async throws -> CachedResponse {
        let started = time.now()
        let response = try await generate(prompt)
        ledger.recordGeneration(nanos: elapsed(since: started))
        return response
    }
}
