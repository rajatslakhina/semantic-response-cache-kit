import Foundation

/// The answer the cache stores and returns.
public struct CachedResponse: Sendable, Hashable {
    public let text: String
    /// Which tools, at which data versions, grounded this response. Empty when
    /// the response was produced from the model's own knowledge alone.
    public let provenance: [ToolProvenance]

    public init(text: String, provenance: [ToolProvenance] = []) {
        self.text = text
        self.provenance = provenance
    }
}

/// A provenance edge: "this response was grounded by tool `tool` when its data
/// was at `version`".
///
/// This is the answer to the staleness problem a semantic cache has that an
/// HTTP cache does not. A cached answer can have been *correct* when a tool
/// call grounded it and be *wrong* now because the tool's data moved — the
/// order shipped, the price changed — while the prompt's embedding is exactly
/// as similar as it was. Nothing about the key tells you. The edge does.
public struct ToolProvenance: Sendable, Hashable {
    public let tool: String
    public let version: String

    public init(tool: String, version: String) {
        self.tool = tool
        self.version = version
    }
}

/// One stored prompt/response pair.
public struct CacheEntry: Sendable, Identifiable, Hashable {
    public let id: UInt64
    /// The normalized prompt text the exact tier keyed on.
    public let normalizedPrompt: String
    public let embedding: Embedding
    public let response: CachedResponse
    public let createdAt: Date
    public private(set) var lastHitAt: Date
    public private(set) var hitCount: Int
    /// Bytes charged against the budget for this entry.
    public let byteCost: Int

    public init(id: UInt64, normalizedPrompt: String, embedding: Embedding,
                response: CachedResponse, createdAt: Date) {
        self.id = id
        self.normalizedPrompt = normalizedPrompt
        self.embedding = embedding
        self.response = response
        self.createdAt = createdAt
        self.lastHitAt = createdAt
        self.hitCount = 0
        self.byteCost = Self.cost(normalizedPrompt: normalizedPrompt, embedding: embedding, response: response)
    }

    mutating func recordHit(at date: Date) {
        lastHitAt = date
        hitCount = Saturating.add(hitCount, 1)
    }

    /// Prompt bytes + response bytes + vector bytes + a flat overhead for the
    /// provenance list. Saturating throughout: a pathological response cannot
    /// overflow the budget arithmetic, it can only fail to fit.
    static func cost(normalizedPrompt: String, embedding: Embedding, response: CachedResponse) -> Int {
        var total = normalizedPrompt.utf8.count
        total = Saturating.add(total, response.text.utf8.count)
        total = Saturating.add(total, Saturating.multiply(embedding.dimension, MemoryLayout<Float>.size))
        for edge in response.provenance {
            total = Saturating.add(total, edge.tool.utf8.count)
            total = Saturating.add(total, edge.version.utf8.count)
        }
        return total
    }
}
