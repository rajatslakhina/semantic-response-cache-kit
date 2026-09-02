/// Turns a raw prompt into the key the exact tier uses.
///
/// This is tier one of the two-tier lookup. It is cheap — no embedding, no NPU
/// — and it exists so that the common case of "the user asked the same thing
/// with different capitalisation and a trailing question mark" never reaches
/// the vector tier at all. Roughly a third of the traffic a semantic cache sees
/// in practice is this kind of near-verbatim repeat, and spending an embedding
/// on it is the first thing a profiler flags.
public struct PromptNormalizer: Sendable {

    public init() {}

    /// Lowercases, strips punctuation, and collapses whitespace.
    ///
    /// Everything that is not a letter or digit acts as a separator. The result
    /// for an input with no letters or digits at all is the empty string; callers
    /// treat that as uncacheable rather than caching under a shared empty key.
    public func normalize(_ prompt: String) -> String {
        var words: [String] = []
        var current = ""
        for scalar in prompt.lowercased().unicodeScalars {
            if scalar.properties.isAlphabetic || scalar.properties.numericType != nil {
                current.unicodeScalars.append(scalar)
            } else if !current.isEmpty {
                words.append(current)
                current = ""
            }
        }
        if !current.isEmpty { words.append(current) }
        return words.joined(separator: " ")
    }

    /// The exact-tier key: a stable hash of the normalized text.
    public func key(for prompt: String) -> UInt64 {
        StableHash.fnv1a(normalize(prompt))
    }

    /// Whitespace-separated tokens of the normalized prompt.
    public func tokens(_ prompt: String) -> [String] {
        let normalized = normalize(prompt)
        guard !normalized.isEmpty else { return [] }
        return normalized.split(separator: " ").map(String.init)
    }
}
