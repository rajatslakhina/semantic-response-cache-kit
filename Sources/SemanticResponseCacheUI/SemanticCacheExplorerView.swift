#if canImport(SwiftUI)
import SwiftUI
import SemanticResponseCache

/// The starting configuration the host app hands the explorer.
///
/// Validated through the same throwing initializers the cache itself uses, so
/// an app cannot launch the explorer into a state the cache would refuse.
public struct ExplorerDefaults: Sendable, Hashable {
    public let threshold: Float
    public let maxEntries: Int
    public let shadowSampleRate: Double
    public let generatorLatencyMilliseconds: Int
    public let useCoverageEviction: Bool

    public init(threshold: Float, maxEntries: Int, shadowSampleRate: Double,
                generatorLatencyMilliseconds: Int, useCoverageEviction: Bool) throws {
        _ = try CachePolicy(similarityThreshold: threshold, maxEntries: maxEntries)
        _ = try ShadowConfiguration(sampleRate: shadowSampleRate)
        guard generatorLatencyMilliseconds >= 0, generatorLatencyMilliseconds <= 5_000 else {
            throw CacheError.invalidBudget("generatorLatencyMilliseconds must be in [0, 5000]")
        }
        self.threshold = threshold
        self.maxEntries = maxEntries
        self.shadowSampleRate = shadowSampleRate
        self.generatorLatencyMilliseconds = generatorLatencyMilliseconds
        self.useCoverageEviction = useCoverageEviction
    }
}

/// Owns the cache, the simulated generator and the replayed trace.
@MainActor
public final class ExplorerModel: ObservableObject {

    public struct PolicyComparison: Sendable, Hashable {
        public let budget: Int
        public let lruHitRate: Double
        public let coverageHitRate: Double
    }

    @Published public var threshold: Float
    @Published public var useCoverageEviction: Bool
    @Published public var includeUnrelated = true
    @Published public private(set) var rows: [ReplayRow] = []
    @Published public private(set) var snapshot: CacheSnapshot?
    @Published public private(set) var breakEven: BreakEven?
    @Published public private(set) var recommendation: ThresholdAdvisor.Recommendation?
    @Published public private(set) var comparison: PolicyComparison?
    @Published public private(set) var isBusy = false
    @Published public private(set) var errorMessage: String?
    @Published public private(set) var ordersVersion = "v1"
    @Published public private(set) var passCount = 0

    private let defaults: ExplorerDefaults
    private var cache: SemanticCache?
    private var generator: SimulatedGenerator?

    public init(defaults: ExplorerDefaults) {
        self.defaults = defaults
        self.threshold = defaults.threshold
        self.useCoverageEviction = defaults.useCoverageEviction
    }

    private var trace: [String] {
        var prompts = DemoCorpus.interleavedTrace()
        if includeUnrelated { prompts.append(contentsOf: DemoCorpus.unrelatedPrompts) }
        return prompts
    }

    /// Builds a fresh cache with the current settings and runs the trace once.
    public func startFresh() async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        errorMessage = nil
        comparison = nil
        do {
            let embedder = try HashedTrigramEmbedder()
            let policy = try CachePolicy(similarityThreshold: threshold, maxEntries: defaults.maxEntries)
            let shadow = try ShadowConfiguration(sampleRate: defaults.shadowSampleRate)
            let eviction: any EvictionPolicy = useCoverageEviction ? CoverageAwareEviction() : LRUEviction()
            let generator = SimulatedGenerator(
                embedder: embedder,
                configuration: .init(latency: .milliseconds(defaults.generatorLatencyMilliseconds)))
            let cache = SemanticCache(policy: policy, embedder: embedder, eviction: eviction, shadow: shadow)
            await cache.registerTool(DemoCorpus.ordersTool, version: "v1")
            await cache.registerTool(DemoCorpus.pricingTool, version: "v1")
            self.cache = cache
            self.generator = generator
            ordersVersion = "v1"
            passCount = 0
            try await replay()
        } catch {
            errorMessage = String(describing: error)
        }
    }

    /// Runs the same trace through the existing cache again — the second pass
    /// is where the hits are.
    public func replayAgain() async {
        guard !isBusy, cache != nil else { return }
        isBusy = true
        defer { isBusy = false }
        do { try await replay() } catch { errorMessage = String(describing: error) }
    }

    /// Simulates the orders API's data moving: the generator now answers with
    /// v2 provenance, and the cache is told so. Every stored answer grounded on
    /// v1 is now stale and will be rejected on its next hit.
    public func bumpOrdersVersion() async {
        guard let cache, let generator, !isBusy else { return }
        let next = ordersVersion == "v1" ? "v2" : "v1"
        await generator.setToolVersion(DemoCorpus.ordersTool, next)
        await cache.registerTool(DemoCorpus.ordersTool, version: next)
        ordersVersion = next
        snapshot = await cache.snapshot()
    }

    /// Same trace, tight budget, both eviction policies — the number the
    /// README's eviction argument rests on, reproduced live.
    public func comparePolicies(budget: Int = 6) async {
        guard !isBusy else { return }
        isBusy = true
        defer { isBusy = false }
        do {
            let lru = try await hitRate(with: LRUEviction(), budget: budget)
            let coverage = try await hitRate(with: CoverageAwareEviction(), budget: budget)
            comparison = PolicyComparison(budget: budget, lruHitRate: lru, coverageHitRate: coverage)
        } catch {
            errorMessage = String(describing: error)
        }
    }

    private func hitRate(with eviction: any EvictionPolicy, budget: Int) async throws -> Double {
        let embedder = try HashedTrigramEmbedder()
        let policy = try CachePolicy(similarityThreshold: threshold, maxEntries: budget)
        let generator = SimulatedGenerator(embedder: embedder)
        let cache = SemanticCache(policy: policy, embedder: embedder, eviction: eviction)
        let prompts = trace
        // Two passes: the first warms, the second is measured.
        _ = try await ReplayRunner.run(prompts, through: cache, generator: generator)
        let rows = try await ReplayRunner.run(prompts, through: cache, generator: generator)
        guard !rows.isEmpty else { return 0 }
        let hits = rows.filter(\.result.isHit).count
        return Double(hits) / Double(rows.count)
    }

    private func replay() async throws {
        guard let cache, let generator else { return }
        rows = try await ReplayRunner.run(trace, through: cache, generator: generator)
        snapshot = await cache.snapshot()
        breakEven = await cache.breakEven()
        recommendation = await cache.recommendation()
        passCount += 1
    }
}

/// The demo screen.
public struct SemanticCacheExplorerView: View {

    @StateObject private var model: ExplorerModel

    public init(defaults: ExplorerDefaults) {
        _model = StateObject(wrappedValue: ExplorerModel(defaults: defaults))
    }

    public var body: some View {
        NavigationStack {
            List {
                policySection
                actionsSection
                if let snapshot = model.snapshot { metricsSection(snapshot) }
                if let comparison = model.comparison { comparisonSection(comparison) }
                if let message = model.errorMessage {
                    Section("Error") { Text(message).font(.footnote).foregroundStyle(.red) }
                }
                traceSection
            }
            .navigationTitle("Semantic Cache")
            .task { if model.rows.isEmpty { await model.startFresh() } }
        }
    }

    private var policySection: some View {
        Section {
            VStack(alignment: .leading) {
                Text("Similarity threshold: \(model.threshold, specifier: "%.2f")")
                Slider(value: $model.threshold, in: 0.30 ... 0.99, step: 0.01)
            }
            Toggle("Coverage-aware eviction (off = LRU)", isOn: $model.useCoverageEviction)
            Toggle("Include unrelated prompts", isOn: $model.includeUnrelated)
        } header: {
            Text("Policy")
        } footer: {
            Text("Threshold and eviction take effect on the next Start fresh.")
        }
    }

    private var actionsSection: some View {
        Section("Actions") {
            Button { Task { await model.startFresh() } } label: {
                Label("Start fresh & run trace", systemImage: "play.fill")
            }
            Button { Task { await model.replayAgain() } } label: {
                Label("Replay same trace (pass \(model.passCount + 1))", systemImage: "arrow.clockwise")
            }
            .disabled(model.snapshot == nil)
            Button { Task { await model.bumpOrdersVersion() } } label: {
                Label("Bump orders-api data version (now \(model.ordersVersion))", systemImage: "shippingbox")
            }
            .disabled(model.snapshot == nil)
            Button { Task { await model.comparePolicies() } } label: {
                Label("Compare LRU vs coverage-aware at budget 6", systemImage: "chart.bar")
            }
        }
        .disabled(model.isBusy)
    }

    private func metricsSection(_ snapshot: CacheSnapshot) -> some View {
        Section("Metrics · \(snapshot.evictionPolicyName) · \(snapshot.entries.count)/\(snapshot.policy.maxEntries) entries") {
            let m = snapshot.metrics
            row("Requests", "\(m.requests)")
            row("Hit rate", m.hitRate.map { Self.percent($0) } ?? "—")
            row("Exact / semantic hits", "\(m.exactHits) / \(m.semanticHits)")
            row("Misses", "\(m.misses)")
            row("Stale rejections", "\(m.staleRejections)")
            row("Evictions", "\(m.evictions)")
            row("Shadow samples", "\(m.shadowSamples) (\(m.shadowDisagreements) disagreed)")
            if let interval = m.falseHitRate {
                row("False-hit rate (95% Wilson)",
                    "\(Self.percent(interval.estimate)) [\(Self.percent(interval.lower))–\(Self.percent(interval.upper))]")
            } else {
                row("False-hit rate", "no shadow samples yet")
            }
            if let breakEven = model.breakEven {
                row("Mean lookup / generation",
                    "\(Self.millis(breakEven.meanLookupNanos)) / \(Self.millis(breakEven.meanGenerationNanos))")
                row("Pays off?", breakEven.paysOff ? "Yes — saves \(Self.millis(breakEven.savedPerRequestNanos))/request"
                                                   : "No — costs \(Self.millis(-breakEven.savedPerRequestNanos))/request")
            }
            if let recommendation = model.recommendation {
                row("Threshold advisor", Self.describe(recommendation))
            }
        }
    }

    private func comparisonSection(_ comparison: ExplorerModel.PolicyComparison) -> some View {
        Section("Eviction comparison · budget \(comparison.budget) entries · second pass") {
            row("LRU hit rate", Self.percent(comparison.lruHitRate))
            row("Coverage-aware hit rate", Self.percent(comparison.coverageHitRate))
        }
    }

    private var traceSection: some View {
        Section("Trace · \(model.rows.count) prompts") {
            if model.rows.isEmpty {
                Text(model.isBusy ? "Running…" : "No trace yet.").foregroundStyle(.secondary)
            }
            ForEach(model.rows) { row in
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        badge(for: row.result)
                        Text(row.prompt).font(.subheadline).lineLimit(2)
                    }
                    Text(row.result.response.text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    if case .semanticHit(let similarity, let matched) = row.result.source {
                        Text("≈ \(similarity, specifier: "%.3f") · matched “\(matched)”")
                            .font(.caption2).foregroundStyle(.tertiary)
                    }
                    if let shadow = row.result.shadow {
                        Text(Self.describe(shadow)).font(.caption2).foregroundStyle(.tertiary)
                    }
                }
            }
        }
    }

    private func badge(for result: CacheResult) -> some View {
        let (text, color): (String, Color)
        switch result.source {
        case .exactHit: (text, color) = ("EXACT", .green)
        case .semanticHit: (text, color) = ("SEMANTIC", .teal)
        case .generated(.coalesced): (text, color) = ("SHARED", .blue)
        case .generated(.stale): (text, color) = ("STALE", .orange)
        case .generated: (text, color) = ("MISS", .gray)
        }
        return Text(text)
            .font(.caption2.weight(.bold))
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.2), in: Capsule())
            .foregroundStyle(color)
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary).multilineTextAlignment(.trailing)
        }
        .font(.footnote)
    }

    private static func percent(_ value: Double) -> String {
        guard value.isFinite else { return "—" }
        return String(format: "%.1f%%", value * 100)
    }

    private static func millis(_ nanos: Double) -> String {
        guard nanos.isFinite else { return "—" }
        return String(format: "%.2f ms", nanos / 1_000_000)
    }

    private static func describe(_ recommendation: ThresholdAdvisor.Recommendation) -> String {
        switch recommendation {
        case .insufficientEvidence(let samples, let needed): return "need \(needed) shadow samples, have \(samples)"
        case .hold: return "hold"
        case .raise(let to): return String(format: "raise to %.2f", to)
        case .considerLowering(let to): return String(format: "could lower to %.2f", to)
        }
    }

    private static func describe(_ verdict: ShadowVerdict) -> String {
        switch verdict {
        case .confirmed: return "shadow: confirmed"
        case .falseHit(let fresh): return "shadow: FALSE HIT → replaced with “\(fresh.text)”"
        case .generatorFailed(let reason): return "shadow: generator failed (\(reason))"
        }
    }
}
#endif
