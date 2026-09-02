# semantic-response-cache-kit

**An on-device semantic response cache for a Foundation Models / Private Cloud Compute / remote inference path — the first cache you will build where a hit can be *wrong*, and there is no server-side gateway to tell you how often.**

[![CI](https://github.com/rajatslakhina/semantic-response-cache-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/rajatslakhina/semantic-response-cache-kit/actions/workflows/ci.yml)
![Swift 6](https://img.shields.io/badge/Swift-6.0-orange) ![iOS 16+](https://img.shields.io/badge/iOS-16%2B-blue) ![Linux](https://img.shields.io/badge/Linux-tested-lightgrey)

Demo app: [semantic-response-cache-kit-demo-app](https://github.com/rajatslakhina/semantic-response-cache-kit-demo-app) — a separate repo that consumes this package as a version-pinned remote dependency.

---

## The problem

Every cache an iOS engineer has shipped keyed on something exact: a URL, an ETag, an object ID. A **semantic** cache breaks that on purpose. The prompt is embedded, and a stored response is returned when cosine similarity clears a threshold. Correctness is no longer a property of the cache — it is a *tuning parameter*, and the false-hit rate is the primary SLO.

The 2026 production guidance for this (threshold ≈ 0.92, working band 0.92–0.97, hit rates of 20–45%, false-hit tolerance of 2% for non-regulated and 0.5% for regulated use) is **server-gateway advice**, and every one of its assumptions inverts on device:

| Server-side assumption | On device |
|---|---|
| The cache saves **dollars** per token. | Foundation Models is zero token cost. The cache saves **latency and battery** — and only if the lookup is cheaper than the generation it skips. That is a measurement, not an assumption. |
| The false-hit rate is computed **offline from gateway logs**. | Nothing leaves the device. The cache has to manufacture its own ground truth. |
| **Cross-user warming** makes cold start short. | One user, one cache. Cold start is permanent, and the hit-rate curve never looks like the benchmarks. |
| Storage is **elastic**; eviction is a policy preference. | Storage is a hard budget. And LRU is *wrong* here, because recency does not predict semantic reuse. |
| A cached answer stays right until it **expires**. | A cached answer was right when a tool call grounded it and is wrong now because the tool's data moved — while the embedding is exactly as similar as it was. |

This package is the system design that follows from taking those five inversions seriously, as runnable Swift 6 code with tests that would fail if the design were faked.

## What's in it

```
Sources/SemanticResponseCache/           (the library — no UI, no app)
  SemanticCache.swift        actor: two-tier lookup, in-flight coalescing, budget, staleness, shadow verification
  Embedding.swift            unit vectors only; refuses NaN / zero / mismatched dimensions
  Embedder.swift             `Embedder` protocol + `HashedTrigramEmbedder` (deterministic, CPU-only, honest about what it is)
  PromptNormalizer.swift     tier 1 key: normalize → FNV-1a (not `Hasher`, which is seeded per process)
  EvictionPolicy.swift       `LRUEviction` and `CoverageAwareEviction` — the one the README argues for, and the baseline it is measured against
  CacheEntry.swift           `CachedResponse` + `ToolProvenance` edges; byte accounting
  ShadowSampler.swift        `ShadowConfiguration`, `SampleDecider` (seeded + system), `ResponseJudge`
  CacheMetrics.swift         counters, `WilsonInterval`, `ThresholdAdvisor`
  CostModel.swift            `LatencyLedger` → `BreakEven`: does this cache pay for itself?
  Saturating.swift           arithmetic that cannot trap (`Int(Double)`, `%`, overflow)
  Fixtures.swift             the support corpus, `SimulatedGenerator`, `ReplayRunner`
Sources/SemanticResponseCacheUI/         (SwiftUI explorer, `#if canImport(SwiftUI)`)
Tests/SemanticResponseCacheTests/        57 XCTest cases, including negative controls
```

### Design decisions, and what was rejected

**1. Two tiers, exact before vector.** `normalize(prompt)` → FNV-1a → dictionary. Only a miss there pays for an embedding. Near-verbatim repeats are a large fraction of real traffic and there is no reason to spend NPU time on a query you could have hashed. *Rejected:* embedding everything and letting a 0.999 similarity absorb exact repeats — it works, and it is the single most expensive way to answer the easiest question.

**2. FNV-1a, not `Hasher`.** Swift's `Hasher` is randomly seeded per process. A key that changes on every launch cannot be persisted, logged, or used to coalesce requests by identity. The exact tier uses a stable 64-bit hash with published test vectors in the suite.

**3. A linear scan, not an ANN index.** For a per-user, on-device cache of hundreds to low thousands of entries a scan is faster than an index's constant factors, has no build step, and is *exact*. An approximate index is a second probabilistic layer on top of the threshold, and one is enough. The limit (~10k entries) is documented in the type, not hidden.

**4. Coverage-aware eviction, with LRU shipped alongside as the control.** LRU evicts the least-recently-hit entry. `CoverageAwareEviction` evicts the entry whose nearest neighbour in the cache is closest to it — the one whose removal loses the least coverage of the prompt space — with ties on fewest hits, then age, then id, so it is total and deterministic. The suite constructs a case where the two policies *must* disagree (two recent near-duplicates and one old singleton) and asserts that they do, and a second test replays the same interleaved trace under a budget of 6 and asserts coverage-aware's second-pass hit rate beats LRU's. On the bundled corpus, threshold 0.50, budget 6: **LRU 3/30 vs coverage-aware 10/30**; at budget 16: **18/30 vs 27/30**. *Rejected:* LFU — hit count is a proxy for coverage, and a worse one; centroid clustering — right idea, more machinery than the entry counts justify.

**5. Provenance edges instead of TTLs.** Every `CachedResponse` carries `[ToolProvenance(tool, version)]`. The cache holds a tool→version registry; `registerTool("orders-api", version: "v2")` makes every entry grounded on v1 *stale*. Stale entries are rejected lazily on their next hit (and the scan continues to the next-best fresh candidate) or swept eagerly with `purgeStale()` / `invalidate(tool:)`. *Rejected:* a global TTL — it expires the answers that are still right and keeps the ones that are wrong for exactly as long as the timer says.

**6. Shadow-mode sampling as the observability design.** A fraction of *semantic* hits also pay for the generation they skipped; a `ResponseJudge` compares the two locally; only the verdict is recorded. Disagreement is a **false hit**, and the cache self-heals: the offending entry leaves, the fresh answer is stored under the paraphrase's own key. The measured rate is reported as a **Wilson interval**, not a point estimate — at k = 0, n = 12 the naive interval says "0% ± 0" and Wilson says the upper bound is ~24%, which is the honest answer. `ThresholdAdvisor` compares the *upper bound* to tolerance and needs 30 samples before it says anything. *Rejected:* comparing embeddings of the two responses as the judge — that is the same instrument grading itself.

**7. Break-even, not hit rate, as the headline metric.** `LatencyLedger` times every lookup and every generation. The cache pays iff `meanLookup < hitRate × meanGeneration`; `BreakEven.breakEvenHitRate` tells you the hit rate you would need. With an embedder slower than the model, the verdict is "no" at *any* hit rate, and the type says so rather than the dashboard showing a green 40%.

**8. In-flight coalescing.** Concurrent identical misses share one generation. This is the same `inFlight` map a request-deduplication layer uses; a cache that lets four identical cold prompts start four generations has not understood its own job.

**9. Per-user isolation, no cross-user warming — stated, not apologised for.** A semantic hit from another user's prompt is a privacy leak with a confidence score attached.

### Actor reentrancy, spelled out

`respond` suspends twice — at the embedder and at the generator — and other callers run in between. After the embedding suspension the exact tier is re-checked (a concurrent caller may have stored the same key). Before storing, the exact index is checked again so a racing insert *replaces* rather than duplicates. `enforceBudget` is a loop that always removes an entry per iteration, and falls back to oldest-first if an `EvictionPolicy` returns something not in the cache, so a policy bug cannot become a stall. `testBudgetInvariantHoldsUnderConcurrentWriters` throws 38 prompts at a 6-entry cache from a task group and asserts the count, the byte total, and one-to-one-ness of the exact index afterwards.

### Crash-safety

No force-unwraps. Every `[i]` is inside a bounds-guarded loop. `Embedding.init(normalizing:)` refuses NaN, infinity, zero norm and empty input, so no degenerate vector can enter the cache. `cosine` returns `nil` on a dimension mismatch rather than reading out of bounds. All counters and byte totals go through `Saturating`. `CachePolicy`, `ShadowConfiguration` and `HashedTrigramEmbedder` throw on invalid parameters at construction, so an invalid configuration cannot reach a request. `Int(Double)` appears nowhere unguarded.

## The embedder is honest

`HashedTrigramEmbedder` is feature hashing over words and padded character trigrams, L2-normalised. It is **deterministic, dependency-free, and runs on Linux**, which is what lets the entire threshold / eviction / staleness / shadow-sampling machinery be exercised on real geometry in CI. It is **not a language model**: it has no notion of synonymy, and on the bundled corpus the same-topic paraphrase similarities run **0.44–0.87** while the worst cross-topic pair sits at **0.55**. Two consequences the package owns rather than hides:

- **The server-side 0.92 is a hit rate of zero with this embedder.** The demo lets you set it and watch. The right threshold is a property of the embedder, which is precisely why the shadow sampler exists.
- **There is no threshold on this corpus with both 100% paraphrase recall and 0 false hits.** Replaying the bundled corpus twice through `ReplayRunner` with shadow sampling at 100%: at 0.45 it measures **3 disagreements in 27 shadow samples** and the advisor says *raise*; at 0.50 it measures **0 in 24**. That overlap is the actual design problem, and the instrument that measures it is the deliverable.

A production deployment substitutes an on-device sentence embedder (`NLEmbedding`, a Core AI model, or a Foundation Models adapter) behind the two-method `Embedder` protocol. Nothing else changes.

## Installation

```swift
dependencies: [
    .package(url: "https://github.com/rajatslakhina/semantic-response-cache-kit.git", from: "1.0.0")
]
// targets: .product(name: "SemanticResponseCache", package: "semantic-response-cache-kit")
//          .product(name: "SemanticResponseCacheUI", package: "semantic-response-cache-kit")  // optional
```

## Usage

```swift
import SemanticResponseCache

let embedder = try HashedTrigramEmbedder()            // or your own `Embedder`
let policy   = try CachePolicy(similarityThreshold: 0.50, maxEntries: 512, maxBytes: 4 << 20,
                               falseHitTolerance: 0.02)
let cache    = SemanticCache(policy: policy, embedder: embedder,
                             eviction: CoverageAwareEviction(),
                             shadow: try ShadowConfiguration(sampleRate: 0.05))

await cache.registerTool("orders-api", version: "2026-09-02T10:00Z")

let result = try await cache.respond(to: "where's my order right now?") { prompt in
    // your real model call, e.g. a LanguageModelSession; return provenance for any tool it used
    CachedResponse(text: answer, provenance: [ToolProvenance(tool: "orders-api", version: currentVersion)])
}

switch result.source {
case .exactHit:                       break
case .semanticHit(let s, let matched): print("served from '\(matched)' at \(s)")
case .generated(let reason):           print("generated:", reason)
}
if case .falseHit? = result.shadow { /* the cache already replaced the entry */ }

let snapshot = await cache.snapshot()
snapshot.metrics.falseHitRate?.upper     // the number to alarm on
await cache.breakEven()?.paysOff         // whether to keep the cache switched on
await cache.recommendation()             // .raise / .hold / .considerLowering / .insufficientEvidence
```

## How to run it

```bash
git clone https://github.com/rajatslakhina/semantic-response-cache-kit.git
cd semantic-response-cache-kit
swift build -Xswiftc -warnings-as-errors
swift test
```

Works on macOS with Xcode 16+ and on Linux with Swift 6.0. The SwiftUI explorer (`SemanticResponseCacheUI`) compiles for iOS 16+ and is what the [demo app](https://github.com/rajatslakhina/semantic-response-cache-kit-demo-app) shows.

## Verification — what actually happened

- **Local (Linux, aarch64, Swift 6.0.3):** clean `rm -rf .build` → `swift build -Xswiftc -warnings-as-errors` → `swift build --build-tests -Xswiftc -warnings-as-errors` → `swift test`: **57 tests, 0 failures**. The numbers quoted above (3/30 vs 10/30, 18/30 vs 27/30, 3-in-27, 0-in-24, 0.44–0.87, 0.55) come from replaying the bundled corpus through the real cache with the same code paths the tests use.
- **CI:** two jobs on every push — a Linux container that repeats the three commands above from a clean checkout, and a `macos-15` job that compiles `SemanticResponseCacheUI` for `generic/platform=iOS Simulator`. See the [Actions tab](https://github.com/rajatslakhina/semantic-response-cache-kit/actions) for the live result rather than a run ID that goes stale.
- **Simulator run:** this repo contains no app. The companion demo app's README states exactly whether it was launched on a Simulator; "compiles for a Simulator" and "ran on a Simulator" are reported there as separate facts.

### Tests that would fail against a faked implementation

- `testExactRepeatIsAnsweredWithoutEmbedding` counts embedder calls — a cache that embeds everything fails it.
- `testBrokenEmbedderIsDetectedByShadowSampler` feeds a `ConstantEmbedder` (every prompt identical) and asserts the Wilson **lower** bound exceeds 0.5 and the advisor says `.raise(to: 0.93)` — a decorative sampler fails it.
- `testShadowCatchesAFalseHitAndSelfHeals` uses a generator whose answer depends on the exact prompt, so every semantic hit is false by construction, and asserts the entry was replaced.
- `testLRUAndCoverageDisagreeOnRedundantRecentEntries` constructs the case where the policies must pick different victims.
- `testCoverageAwareBeatsLRUOnSecondPassUnderTightBudget` is an inequality on a fixed trace, not a magic number.
- `testConcurrentIdenticalMissesShareOneGeneration` uses a 150 ms generator and four real concurrent callers; asserts one call.
- `testBudgetInvariantHoldsUnderConcurrentWriters` has real concurrent writers.
- `testMatchesPublishedFNV1aVectors` pins the exact-tier key to the published FNV-1a constants.
- `testZeroSuccessesHasHonestUpperBound` checks Wilson at k = 0, n = 12 against a hand computation (0.2425).

## Non-goals

Persistence (the entry type is `Hashable` and flat; a store is a one-file addition), an ANN index, and any real model call. The Foundation Models / Core AI adapter is deliberately left to the app: importing either would raise the deployment target above what a cache library has any business requiring.

## License

MIT — see [LICENSE](LICENSE).
