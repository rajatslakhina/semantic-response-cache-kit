import Foundation

/// A small, realistic support corpus: topics, each with a grounded answer and
/// several paraphrases. Shared by the tests and the demo app so that both
/// exercise the same geometry.
public struct DemoTopic: Sendable, Hashable, Identifiable {
    public var id: String { name }
    public let name: String
    public let canonicalPrompt: String
    public let paraphrases: [String]
    public let answer: String
    /// Which tool grounds this answer, or `nil` for pure model knowledge.
    public let groundingTool: String?

    public init(name: String, canonicalPrompt: String, paraphrases: [String], answer: String, groundingTool: String?) {
        self.name = name
        self.canonicalPrompt = canonicalPrompt
        self.paraphrases = paraphrases
        self.answer = answer
        self.groundingTool = groundingTool
    }
}

public enum DemoCorpus {

    public static let ordersTool = "orders-api"
    public static let pricingTool = "pricing-api"

    public static let topics: [DemoTopic] = [
        DemoTopic(name: "order-status",
                  canonicalPrompt: "Where is my order?",
                  paraphrases: ["where is my order", "Where's my order right now?", "Can you tell me where my order is",
                                "what is the status of my order", "Where is my order, has it shipped?"],
                  answer: "Your order #4821 shipped and is out for delivery today.",
                  groundingTool: ordersTool),
        DemoTopic(name: "cancel-order",
                  canonicalPrompt: "How do I cancel my order?",
                  paraphrases: ["how can I cancel my order", "How do I cancel an order I placed", "cancel my order please how"],
                  answer: "Open the order, tap Cancel Order. Orders that have shipped can be returned instead.",
                  groundingTool: ordersTool),
        DemoTopic(name: "refund-timing",
                  canonicalPrompt: "When will I get my refund?",
                  paraphrases: ["when will my refund arrive", "When do I get my refund?", "how long until I get the refund"],
                  answer: "Refunds post to the original payment method within 5–7 business days of the return scanning in.",
                  groundingTool: nil),
        DemoTopic(name: "return-policy",
                  canonicalPrompt: "What is your return policy?",
                  paraphrases: ["what's the return policy", "Tell me about your return policy", "return policy?"],
                  answer: "Most items can be returned within 90 days with a receipt. Some categories have shorter windows.",
                  groundingTool: nil),
        DemoTopic(name: "price-match",
                  canonicalPrompt: "Do you price match?",
                  paraphrases: ["do you do price matching", "Do you price match competitors?", "price match policy"],
                  answer: "Yes — we match identical in-stock items from qualifying retailers at the time of purchase.",
                  groundingTool: pricingTool),
        DemoTopic(name: "delivery-fee",
                  canonicalPrompt: "How much is delivery?",
                  paraphrases: ["how much does delivery cost", "what is the delivery fee", "How much is the delivery charge?"],
                  answer: "Standard delivery is $8.99, free on orders over $45.",
                  groundingTool: pricingTool),
        DemoTopic(name: "store-hours",
                  canonicalPrompt: "What time does the store close?",
                  paraphrases: ["what time do you close", "When does the store close today?", "store closing time"],
                  answer: "Most stores close at 10 PM; the store page shows local hours.",
                  groundingTool: nil),
        DemoTopic(name: "change-address",
                  canonicalPrompt: "Can I change the delivery address?",
                  paraphrases: ["can I change my delivery address", "Is it possible to change the delivery address", "change shipping address"],
                  answer: "Address changes are possible until the order enters packing. After that, contact support.",
                  groundingTool: ordersTool)
    ]

    /// Every paraphrase of every topic, interleaved so that consecutive
    /// prompts are usually different topics — a trace that looks like real
    /// traffic rather than a topic-by-topic drill.
    public static func interleavedTrace() -> [String] {
        var result: [String] = []
        let maxDepth = topics.map(\.paraphrases.count).max() ?? 0
        var depth = 0
        while depth < maxDepth {
            for topic in topics where depth < topic.paraphrases.count {
                result.append(topic.paraphrases[depth])
            }
            depth += 1
        }
        return result
    }

    /// Prompts that belong to none of the topics; a cache should miss on all
    /// of them, and a cache that hits on them has a threshold problem.
    public static let unrelatedPrompts: [String] = [
        "Write a haiku about autumn",
        "What is the capital of Mongolia",
        "Convert 5 kilometers to miles",
        "Recommend a good sci-fi novel"
    ]
}

/// A deterministic stand-in for the model.
///
/// Routes a prompt to the nearest topic by embedding similarity and returns
/// that topic's answer, stamped with the current version of its grounding tool.
/// Below `routingFloor` it answers "I don't know" — like a real model, it does
/// not have an answer for everything, and a cache that stores "I don't know"
/// under a semantic key is doing something a reviewer should notice.
///
/// It is an actor because the demo mutates tool versions at runtime, and the
/// tests need to count how many times it was actually called.
public actor SimulatedGenerator {

    public struct Configuration: Sendable {
        /// Artificial latency per generation. Zero in tests; a few hundred
        /// milliseconds in the demo so the saved latency is visible.
        public var latency: Duration
        /// Minimum similarity to route to a topic.
        public var routingFloor: Float

        public init(latency: Duration = .zero, routingFloor: Float = 0.30) {
            self.latency = latency
            self.routingFloor = routingFloor
        }
    }

    public static let unknownAnswer = "I'm not able to help with that."

    private let embedder: HashedTrigramEmbedder
    private let topics: [(topic: DemoTopic, embedding: Embedding)]
    private let configuration: Configuration
    private var toolVersions: [String: String]
    public private(set) var callCount = 0

    public init(embedder: HashedTrigramEmbedder,
                topics: [DemoTopic] = DemoCorpus.topics,
                toolVersions: [String: String] = [DemoCorpus.ordersTool: "v1", DemoCorpus.pricingTool: "v1"],
                configuration: Configuration = Configuration()) {
        self.embedder = embedder
        self.configuration = configuration
        self.toolVersions = toolVersions
        // Topics whose canonical prompt cannot be embedded are dropped rather
        // than crashing; the corpus above cannot trigger this.
        self.topics = topics.compactMap { topic in
            guard let embedding = try? embedder.embedSynchronously(topic.canonicalPrompt) else { return nil }
            return (topic, embedding)
        }
    }

    public func setToolVersion(_ tool: String, _ version: String) {
        toolVersions[tool] = version
    }

    public func toolVersion(_ tool: String) -> String? { toolVersions[tool] }

    public func generate(_ prompt: String) async throws -> CachedResponse {
        callCount = Saturating.add(callCount, 1)
        if configuration.latency > .zero {
            try await Task.sleep(for: configuration.latency)
        }
        guard let query = try? embedder.embedSynchronously(prompt) else {
            return CachedResponse(text: Self.unknownAnswer)
        }
        var best: (topic: DemoTopic, similarity: Float)?
        for candidate in topics {
            guard let similarity = candidate.embedding.cosine(query) else { continue }
            if let current = best {
                if similarity > current.similarity { best = (candidate.topic, similarity) }
            } else {
                best = (candidate.topic, similarity)
            }
        }
        guard let best, best.similarity >= configuration.routingFloor else {
            return CachedResponse(text: Self.unknownAnswer)
        }
        var provenance: [ToolProvenance] = []
        if let tool = best.topic.groundingTool {
            provenance.append(ToolProvenance(tool: tool, version: toolVersions[tool] ?? "unknown"))
        }
        return CachedResponse(text: best.topic.answer, provenance: provenance)
    }

    /// The generator's own opinion of which topic a prompt belongs to; the
    /// demo uses it to label rows and the tests use it to build ground truth.
    public func route(_ prompt: String) -> DemoTopic? {
        guard let query = try? embedder.embedSynchronously(prompt) else { return nil }
        var best: (topic: DemoTopic, similarity: Float)?
        for candidate in topics {
            guard let similarity = candidate.embedding.cosine(query) else { continue }
            if let current = best {
                if similarity > current.similarity { best = (candidate.topic, similarity) }
            } else {
                best = (candidate.topic, similarity)
            }
        }
        guard let best, best.similarity >= configuration.routingFloor else { return nil }
        return best.topic
    }
}

/// One row of a replayed trace.
public struct ReplayRow: Sendable, Hashable, Identifiable {
    public let id: Int
    public let prompt: String
    public let result: CacheResult
    public let lookupNanos: UInt64

    public init(id: Int, prompt: String, result: CacheResult, lookupNanos: UInt64) {
        self.id = id
        self.prompt = prompt
        self.result = result
        self.lookupNanos = lookupNanos
    }
}

/// Runs a prompt trace through a cache and collects what happened.
public enum ReplayRunner {

    public static func run(_ prompts: [String],
                           through cache: SemanticCache,
                           generator: SimulatedGenerator,
                           timeSource: any TimeSource = MonotonicTimeSource()) async throws -> [ReplayRow] {
        var rows: [ReplayRow] = []
        rows.reserveCapacity(prompts.count)
        var index = 0
        for prompt in prompts {
            let started = timeSource.now()
            let result = try await cache.respond(to: prompt) { text in
                try await generator.generate(text)
            }
            let finished = timeSource.now()
            rows.append(ReplayRow(id: index, prompt: prompt, result: result,
                                  lookupNanos: finished >= started ? finished - started : 0))
            index = Saturating.add(index, 1)
        }
        return rows
    }
}
