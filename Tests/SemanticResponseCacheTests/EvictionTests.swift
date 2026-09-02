import XCTest
@testable import SemanticResponseCache

final class EvictionTests: XCTestCase {

    func testEmptyAndSingletonCases() throws {
        let embedder = try Fixtures.embedder()
        XCTAssertNil(LRUEviction().victim(among: []))
        XCTAssertNil(CoverageAwareEviction().victim(among: []))
        let only = try Fixtures.entry(id: 7, prompt: "hello there", embedder: embedder, lastHit: Date(timeIntervalSince1970: 0))
        XCTAssertEqual(LRUEviction().victim(among: [only]), 7)
        XCTAssertEqual(CoverageAwareEviction().victim(among: [only]), 7)
    }

    /// Three entries: two near-duplicates (A, A') that are recent and one
    /// unrelated entry (B) that is old. LRU evicts B — the only entry covering
    /// its region. Coverage-aware evicts one of the duplicates. The two
    /// policies must disagree here, or the README's argument is empty.
    func testLRUAndCoverageDisagreeOnRedundantRecentEntries() throws {
        let embedder = try Fixtures.embedder()
        let old = Date(timeIntervalSince1970: 1_000)
        let recent = Date(timeIntervalSince1970: 2_000)
        let a = try Fixtures.entry(id: 1, prompt: "where is my order", embedder: embedder, lastHit: recent, hits: 5)
        let aPrime = try Fixtures.entry(id: 2, prompt: "where is my order right now", embedder: embedder, lastHit: recent, hits: 5)
        let b = try Fixtures.entry(id: 3, prompt: "write a haiku about autumn", embedder: embedder, lastHit: old, hits: 1)

        XCTAssertEqual(LRUEviction().victim(among: [a, aPrime, b]), 3)
        let coverageVictim = CoverageAwareEviction().victim(among: [a, aPrime, b])
        XCTAssertTrue(coverageVictim == 1 || coverageVictim == 2, "coverage-aware must evict a duplicate, got \(String(describing: coverageVictim))")
    }

    func testCoverageTieBreaksOnHitsThenAgeThenID() throws {
        let embedder = try Fixtures.embedder()
        let t0 = Date(timeIntervalSince1970: 0)
        let t1 = Date(timeIntervalSince1970: 1)
        // Two identical-redundancy pairs (each entry's nearest neighbour is its
        // twin); vary hits and age to force each tie-break in turn.
        let a = try Fixtures.entry(id: 1, prompt: "where is my order", embedder: embedder, lastHit: t1, hits: 2)
        let aPrime = try Fixtures.entry(id: 2, prompt: "where is my order", embedder: embedder, lastHit: t1, hits: 1)
        XCTAssertEqual(CoverageAwareEviction().victim(among: [a, aPrime]), 2, "fewer hits loses")

        let b = try Fixtures.entry(id: 3, prompt: "where is my order", embedder: embedder, lastHit: t0, hits: 1)
        let bPrime = try Fixtures.entry(id: 4, prompt: "where is my order", embedder: embedder, lastHit: t1, hits: 1)
        XCTAssertEqual(CoverageAwareEviction().victim(among: [b, bPrime]), 3, "older loses")

        let c = try Fixtures.entry(id: 5, prompt: "where is my order", embedder: embedder, lastHit: t1, hits: 1)
        let cPrime = try Fixtures.entry(id: 6, prompt: "where is my order", embedder: embedder, lastHit: t1, hits: 1)
        XCTAssertEqual(CoverageAwareEviction().victim(among: [c, cPrime]), 5, "lower id loses")
        XCTAssertEqual(CoverageAwareEviction().victim(among: [cPrime, c]), 5, "order of input must not matter")
    }

    func testLRUTieBreaksOnID() throws {
        let embedder = try Fixtures.embedder()
        let t = Date(timeIntervalSince1970: 5)
        let a = try Fixtures.entry(id: 9, prompt: "one", embedder: embedder, lastHit: t)
        let b = try Fixtures.entry(id: 4, prompt: "two", embedder: embedder, lastHit: t)
        XCTAssertEqual(LRUEviction().victim(among: [a, b]), 4)
    }

    /// The claim the demo makes live: under a budget too small for the corpus,
    /// coverage-aware eviction retains more distinct topics than LRU on the
    /// second pass of an interleaved trace. Asserted as an inequality on a
    /// fixed trace, not as a magic number.
    func testCoverageAwareBeatsLRUOnSecondPassUnderTightBudget() async throws {
        func secondPassHitRate(_ eviction: any EvictionPolicy) async throws -> Double {
            let embedder = try Fixtures.embedder()
            let cache = try Fixtures.cache(threshold: 0.60, maxEntries: 6, embedder: embedder, eviction: eviction)
            let generator = SimulatedGenerator(embedder: embedder)
            let trace = DemoCorpus.interleavedTrace() + DemoCorpus.unrelatedPrompts
            _ = try await ReplayRunner.run(trace, through: cache, generator: generator)
            let rows = try await ReplayRunner.run(trace, through: cache, generator: generator)
            return Double(rows.filter(\.result.isHit).count) / Double(rows.count)
        }
        let lru = try await secondPassHitRate(LRUEviction())
        let coverage = try await secondPassHitRate(CoverageAwareEviction())
        XCTAssertGreaterThan(coverage, lru, "coverage \(coverage) vs LRU \(lru)")
    }
}
