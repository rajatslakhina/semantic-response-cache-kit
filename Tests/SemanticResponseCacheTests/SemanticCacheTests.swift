import XCTest
@testable import SemanticResponseCache

final class SemanticCacheTests: XCTestCase {

    // MARK: Two tiers

    func testExactRepeatIsAnsweredWithoutEmbedding() async throws {
        let counting = CountingEmbedder(try Fixtures.embedder())
        let cache = try Fixtures.cache(embedder: counting)
        let generator = SimulatedGenerator(embedder: try Fixtures.embedder())

        let first = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        let second = try await cache.respond(to: "  where IS my order  ") { try await generator.generate($0) }

        XCTAssertEqual(first.source, .generated(.cold))
        XCTAssertEqual(second.source, .exactHit)
        XCTAssertEqual(second.response, first.response)
        XCTAssertEqual(counting.calls, 1, "the exact tier must answer the repeat without touching the embedder")
        let calls = await generator.callCount
        XCTAssertEqual(calls, 1)
    }

    func testParaphraseIsASemanticHitAndUnrelatedIsNot() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)

        _ = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        let hit = try await cache.respond(to: "where's my order right now?") { try await generator.generate($0) }
        let miss = try await cache.respond(to: "Write a haiku about autumn") { try await generator.generate($0) }

        guard case .semanticHit(let similarity, let matched) = hit.source else {
            return XCTFail("expected semantic hit, got \(hit.source)")
        }
        // Compare the reported similarity against a cosine computed outside the
        // cache, so the assertion can fail if the cache reports the wrong number.
        let expectedHit = try XCTUnwrap(embedder.embedSynchronously("Where is my order?")
            .cosine(embedder.embedSynchronously("where's my order right now?")))
        XCTAssertEqual(similarity, expectedHit, accuracy: 1e-6)
        XCTAssertGreaterThanOrEqual(expectedHit, 0.55, "precondition: this paraphrase must clear the threshold")
        XCTAssertEqual(matched, "where is my order")
        XCTAssertTrue(hit.isHit)

        guard case .generated(.belowThreshold(let nearest)) = miss.source else {
            return XCTFail("expected below-threshold miss, got \(miss.source)")
        }
        // The paraphrase was served, not stored, so the only entry the unrelated
        // prompt was compared against is the canonical one; `nearest` must equal
        // that cosine exactly.
        let haiku = try embedder.embedSynchronously("Write a haiku about autumn")
        let expectedNearest = try XCTUnwrap(embedder.embedSynchronously("Where is my order?").cosine(haiku))
        XCTAssertEqual(nearest, expectedNearest, accuracy: 1e-6)
        XCTAssertLessThan(expectedNearest, 0.55, "precondition: the unrelated prompt must miss")
        XCTAssertFalse(miss.isHit)

        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.semanticHits, 1)
        XCTAssertEqual(snapshot.metrics.misses, 2)
        XCTAssertEqual(snapshot.metrics.nearMissSimilarities.count, 1)
        XCTAssertEqual(snapshot.entries.count, 2)
    }

    func testThresholdOfOneNeverProducesASemanticHit() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 1.0, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)
        for prompt in DemoCorpus.topics[0].paraphrases {
            let result = try await cache.respond(to: prompt) { try await generator.generate($0) }
            if case .semanticHit = result.source { XCTFail("threshold 1.0 must not semantic-hit: \(prompt)") }
        }
    }

    // MARK: Uncacheable inputs

    func testEmptyPromptIsAnsweredButNotStored() async throws {
        let cache = try Fixtures.cache(embedder: try Fixtures.embedder())
        let result = try await cache.respond(to: "???") { _ in CachedResponse(text: "shrug") }
        XCTAssertEqual(result.source, .generated(.uncacheablePrompt))
        XCTAssertEqual(result.response.text, "shrug")
        let count = await cache.count
        XCTAssertEqual(count, 0)
        let metrics = await cache.snapshot().metrics
        XCTAssertEqual(metrics.uncacheable, 1)
    }

    func testEmbedderFailureStillAnswersAndStoresNothing() async throws {
        let cache = try Fixtures.cache(embedder: FailingEmbedder())
        let result = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "ok") }
        XCTAssertEqual(result.source, .generated(.embeddingFailed))
        XCTAssertEqual(result.response.text, "ok")
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }

    func testWrongDimensionEmbedderIsTreatedAsEmbeddingFailure() async throws {
        let cache = try Fixtures.cache(embedder: WrongDimensionEmbedder())
        let result = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "ok") }
        XCTAssertEqual(result.source, .generated(.embeddingFailed))
        let count = await cache.count
        XCTAssertEqual(count, 0)
    }

    func testGeneratorErrorPropagatesAndLeavesNoInFlightGhost() async throws {
        struct Boom: Error {}
        let cache = try Fixtures.cache(embedder: try Fixtures.embedder())
        do {
            _ = try await cache.respond(to: "Where is my order?") { _ in throw Boom() }
            XCTFail("expected throw")
        } catch is Boom {
            // expected
        }
        // A retry must run the generator again rather than await a dead task.
        let retry = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "recovered") }
        XCTAssertEqual(retry.response.text, "recovered")
        XCTAssertEqual(retry.source, .generated(.cold))
    }

    // MARK: Budget

    func testEntryBudgetIsEnforcedWithEvictions() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.99, maxEntries: 5, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)
        let prompts = DemoCorpus.topics.map(\.canonicalPrompt) // 8 distinct
        for prompt in prompts {
            _ = try await cache.respond(to: prompt) { try await generator.generate($0) }
        }
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.entries.count, 5)
        XCTAssertEqual(snapshot.metrics.evictions, 3)
    }

    func testResponseLargerThanByteBudgetIsNotStored() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(maxBytes: 300, embedder: embedder)
        let huge = String(repeating: "x", count: 1_000)
        let result = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: huge) }
        XCTAssertEqual(result.response.text, huge)
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.entries.count, 0)
        XCTAssertEqual(snapshot.totalBytes, 0)
        XCTAssertEqual(snapshot.metrics.uncacheable, 1)
    }

    func testByteBudgetEvictsUntilUnderLimit() async throws {
        let embedder = try Fixtures.embedder()
        // Each entry costs ~1 KB of vector (256 × 4) plus text; allow ~2.5 entries.
        let cache = try Fixtures.cache(threshold: 0.99, maxBytes: 2_700, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)
        for prompt in DemoCorpus.topics.map(\.canonicalPrompt) {
            _ = try await cache.respond(to: prompt) { try await generator.generate($0) }
        }
        let snapshot = await cache.snapshot()
        XCTAssertLessThanOrEqual(snapshot.totalBytes, 2_700)
        XCTAssertEqual(snapshot.totalBytes, snapshot.entries.reduce(0) { $0 + $1.byteCost })
        XCTAssertGreaterThan(snapshot.metrics.evictions, 0)
        XCTAssertGreaterThan(snapshot.entries.count, 0)
    }

    func testMisbehavingEvictionPolicyCannotStallTheBudget() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.99, maxEntries: 2, embedder: embedder, eviction: BogusEviction())
        let generator = SimulatedGenerator(embedder: embedder)
        for prompt in DemoCorpus.topics.map(\.canonicalPrompt) {
            _ = try await cache.respond(to: prompt) { try await generator.generate($0) }
        }
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.entries.count, 2)
        // Fallback is oldest-first, so the two survivors are the two newest.
        XCTAssertEqual(snapshot.entries.map(\.normalizedPrompt),
                       ["what time does the store close", "can i change the delivery address"])
    }

    // MARK: Staleness / provenance

    func testToolVersionBumpMakesGroundedEntryStale() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.60, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)
        await cache.registerTool(DemoCorpus.ordersTool, version: "v1")

        let first = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        XCTAssertEqual(first.response.provenance, [ToolProvenance(tool: DemoCorpus.ordersTool, version: "v1")])

        // Data moves.
        await generator.setToolVersion(DemoCorpus.ordersTool, "v2")
        await cache.registerTool(DemoCorpus.ordersTool, version: "v2")

        // Exact tier: stale entry rejected, regenerated, re-stored at v2.
        let exact = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        XCTAssertEqual(exact.source, .generated(.stale))
        XCTAssertEqual(exact.response.provenance.first?.version, "v2")

        let again = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        XCTAssertEqual(again.source, .exactHit)

        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.staleRejections, 1)
        XCTAssertEqual(snapshot.entries.count, 1)
        let calls = await generator.callCount
        XCTAssertEqual(calls, 2)
    }

    func testStaleEntryOnVectorTierIsSkippedForNextFreshCandidate() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder)
        await cache.registerTool("t", version: "v1")
        // Two entries far enough apart to both be stored, and a compound query
        // that clears the threshold against both. The closer one is grounded on
        // a tool that then moves.
        let grounded = "when will my refund arrive"
        let ungrounded = "where is my order"
        let query = "where is my order and when will my refund arrive"
        let a = try embedder.embedSynchronously(grounded)
        let b = try embedder.embedSynchronously(ungrounded)
        let q = try embedder.embedSynchronously(query)
        XCTAssertLessThan(try XCTUnwrap(a.cosine(b)), 0.55, "precondition: the two entries must not hit each other")
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(a.cosine(q)), 0.55, "precondition")
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(b.cosine(q)), 0.55, "precondition")
        XCTAssertGreaterThan(try XCTUnwrap(a.cosine(q)), try XCTUnwrap(b.cosine(q)), "precondition: grounded is the closer candidate")

        _ = try await cache.respond(to: grounded) { _ in
            CachedResponse(text: "grounded", provenance: [ToolProvenance(tool: "t", version: "v1")])
        }
        _ = try await cache.respond(to: ungrounded) { _ in CachedResponse(text: "ungrounded") }
        let stored = await cache.count
        XCTAssertEqual(stored, 2)
        await cache.registerTool("t", version: "v2")

        let result = try await cache.respond(to: query) { _ in CachedResponse(text: "fresh") }
        // The grounded (closer) entry was stale and discarded; the ungrounded
        // one is still above threshold and answers.
        guard case .semanticHit(_, let matched) = result.source else {
            return XCTFail("expected a semantic hit from the surviving entry, got \(result.source)")
        }
        XCTAssertEqual(matched, ungrounded)
        XCTAssertEqual(result.response.text, "ungrounded")
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.staleRejections, 1)
        XCTAssertEqual(snapshot.entries.count, 1)
    }

    func testUnknownToolInRegistryDoesNotMakeEntriesStale() async throws {
        let cache = try Fixtures.cache(embedder: try Fixtures.embedder())
        _ = try await cache.respond(to: "Where is my order?") { _ in
            CachedResponse(text: "a", provenance: [ToolProvenance(tool: "never-registered", version: "v9")])
        }
        let result = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "b") }
        XCTAssertEqual(result.source, .exactHit)
        XCTAssertEqual(result.response.text, "a")
    }

    func testEagerInvalidateAndPurge() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.99, embedder: embedder)
        let generator = SimulatedGenerator(embedder: embedder)
        await cache.registerTool(DemoCorpus.ordersTool, version: "v1")
        await cache.registerTool(DemoCorpus.pricingTool, version: "v1")
        for prompt in DemoCorpus.topics.map(\.canonicalPrompt) {
            _ = try await cache.respond(to: prompt) { try await generator.generate($0) }
        }
        var count = await cache.count
        XCTAssertEqual(count, 8)

        let ordersGrounded = DemoCorpus.topics.filter { $0.groundingTool == DemoCorpus.ordersTool }.count
        let removed = await cache.invalidate(tool: DemoCorpus.ordersTool)
        XCTAssertEqual(removed, ordersGrounded)
        count = await cache.count
        XCTAssertEqual(count, 8 - ordersGrounded)

        await cache.registerTool(DemoCorpus.pricingTool, version: "v2")
        let pricingGrounded = DemoCorpus.topics.filter { $0.groundingTool == DemoCorpus.pricingTool }.count
        let purged = await cache.purgeStale()
        XCTAssertEqual(purged, pricingGrounded)
        count = await cache.count
        XCTAssertEqual(count, 8 - ordersGrounded - pricingGrounded)
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.totalBytes, snapshot.entries.reduce(0) { $0 + $1.byteCost })
    }

    // MARK: Concurrency

    func testConcurrentIdenticalMissesShareOneGeneration() async throws {
        let cache = try Fixtures.cache(embedder: try Fixtures.embedder())
        let generator = SlowConstantGenerator(delay: .milliseconds(150))
        let results = try await withThrowingTaskGroup(of: CacheResult.self, returning: [CacheResult].self) { group in
            for _ in 0 ..< 4 {
                group.addTask {
                    try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
                }
            }
            var collected: [CacheResult] = []
            for try await result in group { collected.append(result) }
            return collected
        }
        let calls = await generator.calls
        XCTAssertEqual(calls, 1, "four identical concurrent misses must produce one generation")
        let coalesced = results.filter { $0.source == .generated(.coalesced) }.count
        let snapshot = await cache.snapshot()
        XCTAssertEqual(coalesced, 3)
        XCTAssertEqual(snapshot.metrics.coalescedRequests, 3)
        XCTAssertEqual(snapshot.entries.count, 1, "the exact index must stay one-to-one under concurrency")
        XCTAssertTrue(results.allSatisfy { $0.response.text == "constant" })
    }

    func testBudgetInvariantHoldsUnderConcurrentWriters() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.99, maxEntries: 6, maxBytes: 20_000, embedder: embedder)
        var prompts: [String] = []
        for topic in DemoCorpus.topics {
            prompts.append(topic.canonicalPrompt)
            prompts.append(contentsOf: topic.paraphrases)
        }
        prompts.append(contentsOf: DemoCorpus.unrelatedPrompts)
        XCTAssertGreaterThan(prompts.count, 30)

        try await withThrowingTaskGroup(of: Void.self) { group in
            for prompt in prompts {
                group.addTask {
                    _ = try await cache.respond(to: prompt) { text in
                        try await Task.sleep(for: .milliseconds(2))
                        return CachedResponse(text: "answer: \(text)")
                    }
                }
            }
            try await group.waitForAll()
        }
        let snapshot = await cache.snapshot()
        XCTAssertLessThanOrEqual(snapshot.entries.count, 6)
        XCTAssertLessThanOrEqual(snapshot.totalBytes, 20_000)
        XCTAssertEqual(snapshot.totalBytes, snapshot.entries.reduce(0) { $0 + $1.byteCost })
        XCTAssertEqual(snapshot.metrics.requests, prompts.count)
        XCTAssertGreaterThan(snapshot.metrics.evictions, 0)
        // Distinct normalized prompts must map to distinct entries — no duplicates.
        XCTAssertEqual(Set(snapshot.entries.map(\.normalizedPrompt)).count, snapshot.entries.count)
    }

    // MARK: Shadow sampling

    func testShadowConfirmsAConsistentGenerator() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder, shadowRate: 1, decider: AlwaysSample())
        let generator = SimulatedGenerator(embedder: embedder)
        _ = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        let hit = try await cache.respond(to: "where's my order right now") { try await generator.generate($0) }
        XCTAssertEqual(hit.shadow, .confirmed)
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.shadowSamples, 1)
        XCTAssertEqual(snapshot.metrics.shadowDisagreements, 0)
        let calls = await generator.callCount
        XCTAssertEqual(calls, 2, "a shadow sample pays for the generation it skipped")
    }

    func testShadowCatchesAFalseHitAndSelfHeals() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder, shadowRate: 1, decider: AlwaysSample())
        let generator = ExactPromptGenerator()
        _ = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        let hit = try await cache.respond(to: "where's my order right now") { try await generator.generate($0) }

        // The caller still received the cached (wrong) answer — shadow mode
        // measures, it does not block. The verdict says so.
        XCTAssertEqual(hit.response.text, "answer for: Where is my order?")
        guard case .falseHit(let fresh)? = hit.shadow else { return XCTFail("expected false-hit verdict, got \(String(describing: hit.shadow))") }
        XCTAssertEqual(fresh.text, "answer for: where's my order right now")

        // Self-heal: the offending entry is gone and the fresh answer is stored
        // under the paraphrase's own key.
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.shadowDisagreements, 1)
        XCTAssertEqual(snapshot.entries.map(\.normalizedPrompt), ["where s my order right now"])
        let exact = try await cache.respond(to: "where's my order right now") { try await generator.generate($0) }
        XCTAssertEqual(exact.source, .exactHit)
        XCTAssertEqual(exact.response.text, "answer for: where's my order right now")
    }

    func testShadowGeneratorFailureIsRecordedNotThrown() async throws {
        struct Boom: Error {}
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder, shadowRate: 1, decider: AlwaysSample())
        _ = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "a") }
        let hit = try await cache.respond(to: "where's my order right now") { _ in throw Boom() }
        guard case .generatorFailed? = hit.shadow else { return XCTFail("expected generatorFailed") }
        XCTAssertEqual(hit.response.text, "a")
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.shadowFailures, 1)
        XCTAssertEqual(snapshot.metrics.shadowSamples, 0)
    }

    func testShadowIsNotSampledAtRateZero() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.55, embedder: embedder, shadowRate: 0, decider: AlwaysSample())
        let generator = ExactPromptGenerator()
        _ = try await cache.respond(to: "Where is my order?") { try await generator.generate($0) }
        let hit = try await cache.respond(to: "where's my order right now") { try await generator.generate($0) }
        XCTAssertNil(hit.shadow)
        let calls = await generator.calls
        XCTAssertEqual(calls, 1)
    }

    /// The negative control for the whole instrument. A broken embedder makes
    /// every prompt a semantic hit of the first; a strict judge plus shadow
    /// sampling must turn that into a measured false-hit rate whose upper bound
    /// blows through tolerance and a `raise` recommendation. If this test
    /// passes with the sampler disabled, the sampler is decorative.
    func testBrokenEmbedderIsDetectedByShadowSampler() async throws {
        let cache = try Fixtures.cache(threshold: 0.92, maxEntries: 256, embedder: ConstantEmbedder(),
                                       shadowRate: 1, decider: AlwaysSample())
        var prompts: [String] = []
        for topic in DemoCorpus.topics {
            prompts.append(topic.canonicalPrompt)
            prompts.append(contentsOf: topic.paraphrases)
        }
        prompts.append(contentsOf: DemoCorpus.unrelatedPrompts)
        let generator = ExactPromptGenerator()
        var semanticHits = 0
        for prompt in prompts {
            let result = try await cache.respond(to: prompt) { try await generator.generate($0) }
            if case .semanticHit = result.source { semanticHits += 1 }
        }
        XCTAssertGreaterThanOrEqual(semanticHits, 30, "the broken embedder should make almost everything a hit")
        let snapshot = await cache.snapshot()
        let interval = try XCTUnwrap(snapshot.metrics.falseHitRate)
        XCTAssertGreaterThanOrEqual(interval.samples, 30)
        XCTAssertGreaterThan(interval.lower, 0.5, "every sampled hit disagreed; the interval must say so")
        XCTAssertEqual(interval.estimate, 1, accuracy: 1e-9)
        let recommendation = await cache.recommendation()
        XCTAssertEqual(recommendation, .raise(to: 0.93))
    }

    func testCorrectEmbedderAndConsistentGeneratorStayInsideTolerance() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(threshold: 0.60, embedder: embedder, shadowRate: 1, decider: AlwaysSample())
        let generator = SimulatedGenerator(embedder: embedder)
        var prompts: [String] = []
        for topic in DemoCorpus.topics { prompts.append(contentsOf: topic.paraphrases) }
        for pass in 0 ..< 2 {
            for prompt in prompts {
                let result = try await cache.respond(to: prompt) { try await generator.generate($0) }
                if pass == 1 { XCTAssertTrue(result.isHit, "second pass must hit: \(prompt)") }
                if case .falseHit? = result.shadow { XCTFail("false hit on \(prompt)") }
            }
        }
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.metrics.shadowDisagreements, 0)
        XCTAssertGreaterThan(snapshot.metrics.shadowSamples, 0)
    }

    // MARK: Cost model

    func testBreakEvenUsesMeasuredLookupAndGenerationTime() async throws {
        // Lookup 1: 100 → miss, generation 1_000. Lookup 2: 100 → exact hit.
        // respond() calls now(): start, [lookup end], [gen start, gen end].
        let time = ScriptedTime(steps: [0, 100, 0, 1_000, 0, 100])
        let cache = try Fixtures.cache(embedder: try Fixtures.embedder(), timeSource: time)
        _ = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "a") }
        _ = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "b") }
        let measured = await cache.breakEven()
        let breakEven = try XCTUnwrap(measured)
        XCTAssertEqual(breakEven.meanLookupNanos, 100, accuracy: 1e-9)
        XCTAssertEqual(breakEven.meanGenerationNanos, 1_000, accuracy: 1e-9)
        XCTAssertEqual(breakEven.hitRate, 0.5, accuracy: 1e-9)
        XCTAssertEqual(breakEven.savedPerRequestNanos, 400, accuracy: 1e-9) // 0.5 × 1000 − 100
        XCTAssertTrue(breakEven.paysOff)
        XCTAssertEqual(try XCTUnwrap(breakEven.breakEvenHitRate), 0.1, accuracy: 1e-9)
    }

    func testBreakEvenReportsALosingCacheAsLosing() {
        var ledger = LatencyLedger()
        ledger.recordLookup(nanos: 900)
        ledger.recordGeneration(nanos: 1_000)
        let verdict = ledger.breakEven(hitRate: 0.5)
        XCTAssertEqual(verdict?.paysOff, false)
        XCTAssertEqual(verdict?.savedPerRequestNanos ?? 0, -400, accuracy: 1e-9)
        XCTAssertNil(ledger.breakEven(hitRate: nil))
        XCTAssertNil(LatencyLedger().breakEven(hitRate: 0.5))
    }

    // MARK: Advisor & metrics bounds

    func testAdvisorNeedsEvidenceThenRaisesHoldsOrLowers() throws {
        let policy = try CachePolicy(similarityThreshold: 0.92, falseHitTolerance: 0.02)
        var metrics = CacheMetrics()
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: metrics, policy: policy),
                       .insufficientEvidence(samples: 0, needed: 30))

        for _ in 0 ..< 29 { metrics.recordShadow(.confirmed) }
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: metrics, policy: policy),
                       .insufficientEvidence(samples: 29, needed: 30))

        // 30 clean samples: Wilson upper ≈ 11.4% > 2% → raise. Zero observed
        // failures is *not* evidence of a 0% rate at n = 30.
        metrics.recordShadow(.confirmed)
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: metrics, policy: policy), .raise(to: 0.93))

        // 2,000 clean samples: upper ≈ 0.19% < 0.5% → could lower.
        var many = CacheMetrics()
        for _ in 0 ..< 2_000 { many.recordShadow(.confirmed) }
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: many, policy: policy), .considerLowering(to: 0.91))

        // 2,000 samples with 20 failures: estimate 1%, upper ≈ 1.54% → hold.
        var mixed = CacheMetrics()
        for _ in 0 ..< 1_980 { mixed.recordShadow(.confirmed) }
        for _ in 0 ..< 20 { mixed.recordShadow(.falseHit(fresh: CachedResponse(text: "x"))) }
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: mixed, policy: policy), .hold)

        // At the ceiling, "raise" degrades to hold rather than exceeding 0.99.
        let ceiling = try policy.withThreshold(0.99)
        XCTAssertEqual(ThresholdAdvisor.recommend(metrics: metrics, policy: ceiling), .hold)
    }

    func testNearMissHistoryIsBounded() {
        var metrics = CacheMetrics()
        for i in 0 ..< 1_000 { metrics.recordMiss(nearest: Float(i) / 1_000) }
        XCTAssertEqual(metrics.nearMissSimilarities.count, CacheMetrics.nearMissHistoryLimit)
        XCTAssertEqual(metrics.nearMissSimilarities.first ?? -1, Float(1_000 - 256) / 1_000, accuracy: 1e-6)
        metrics.recordMiss(nearest: .nan)
        XCTAssertEqual(metrics.nearMissSimilarities.count, CacheMetrics.nearMissHistoryLimit)
    }

    func testRemoveAllResetsStorageButKeepsMetrics() async throws {
        let embedder = try Fixtures.embedder()
        let cache = try Fixtures.cache(embedder: embedder)
        _ = try await cache.respond(to: "Where is my order?") { _ in CachedResponse(text: "a") }
        await cache.removeAll()
        let snapshot = await cache.snapshot()
        XCTAssertEqual(snapshot.entries.count, 0)
        XCTAssertEqual(snapshot.totalBytes, 0)
        XCTAssertEqual(snapshot.metrics.requests, 1)
    }
}
