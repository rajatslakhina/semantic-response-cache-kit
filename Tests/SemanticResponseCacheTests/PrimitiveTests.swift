import XCTest
@testable import SemanticResponseCache

final class SaturatingTests: XCTestCase {

    func testAddClampsInsteadOfTrapping() {
        XCTAssertEqual(Saturating.add(Int.max, 1), Int.max)
        XCTAssertEqual(Saturating.add(Int.min, -1), Int.min)
        XCTAssertEqual(Saturating.add(40, 2), 42)
    }

    func testMultiplyClampsInsteadOfTrapping() {
        XCTAssertEqual(Saturating.multiply(Int.max, 2), Int.max)
        XCTAssertEqual(Saturating.multiply(Int.max, -2), Int.min)
        XCTAssertEqual(Saturating.multiply(Int.min, -1), Int.max)
        XCTAssertEqual(Saturating.multiply(6, 7), 42)
    }

    func testIntConversionHandlesEveryTrappingInput() {
        XCTAssertEqual(Saturating.int(.nan, fallback: -7), -7)
        XCTAssertEqual(Saturating.int(.infinity), Int.max)
        XCTAssertEqual(Saturating.int(-.infinity), Int.min)
        XCTAssertEqual(Saturating.int(1e300), Int.max)
        XCTAssertEqual(Saturating.int(-1e300), Int.min)
        XCTAssertEqual(Saturating.int(Double(Int.max)), Int.max)
        XCTAssertEqual(Saturating.int(3.9), 3)
    }

    func testDivideHandlesZeroAndMinByMinusOne() {
        XCTAssertEqual(Saturating.divide(10, by: 0, fallback: 99), 99)
        XCTAssertEqual(Saturating.divide(Int.min, by: -1), Int.max)
        XCTAssertEqual(Saturating.divide(9, by: 3), 3)
    }
}

final class StableHashTests: XCTestCase {

    /// Published FNV-1a 64-bit test vectors. If these fail, the exact tier's
    /// keys are not the FNV-1a they claim to be — and a persisted cache would
    /// not survive an upgrade.
    func testMatchesPublishedFNV1aVectors() {
        XCTAssertEqual(StableHash.fnv1a(""), 0xcbf2_9ce4_8422_2325)
        XCTAssertEqual(StableHash.fnv1a("a"), 0xaf63_dc4c_8601_ec8c)
        XCTAssertEqual(StableHash.fnv1a("foobar"), 0x85944171f73967e8)
    }

    func testByteAndStringOverloadsAgree() {
        let text = "where is my order"
        XCTAssertEqual(StableHash.fnv1a(text), StableHash.fnv1a(bytes: Array(text.utf8)))
    }
}

final class PromptNormalizerTests: XCTestCase {
    let normalizer = PromptNormalizer()

    func testCollapsesCaseWhitespaceAndPunctuation() {
        XCTAssertEqual(normalizer.normalize("  Where's   my ORDER?! "), "where s my order")
        XCTAssertEqual(normalizer.normalize("Order #4821, status."), "order 4821 status")
    }

    func testPunctuationOnlyPromptNormalizesToEmpty() {
        XCTAssertEqual(normalizer.normalize("?!... ---"), "")
        XCTAssertEqual(normalizer.tokens("?!... ---"), [])
    }

    func testKeyIsStableAcrossVariants() {
        XCTAssertEqual(normalizer.key(for: "Where is my order?"), normalizer.key(for: "where IS my order"))
        XCTAssertNotEqual(normalizer.key(for: "Where is my order?"), normalizer.key(for: "Where is my refund?"))
    }
}

final class EmbeddingTests: XCTestCase {

    func testRefusesDegenerateVectors() {
        XCTAssertNil(Embedding(normalizing: []))
        XCTAssertNil(Embedding(normalizing: [0, 0, 0]))
        XCTAssertNil(Embedding(normalizing: [1, .nan]))
        XCTAssertNil(Embedding(normalizing: [1, .infinity]))
    }

    func testNormalizesToUnitLength() throws {
        let embedding = try XCTUnwrap(Embedding(normalizing: [3, 4]))
        XCTAssertEqual(embedding.values[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(embedding.values[1], 0.8, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(embedding.cosine(embedding)), 1, accuracy: 1e-6)
    }

    func testCosineOfOrthogonalAndOppositeVectors() throws {
        let x = try XCTUnwrap(Embedding(normalizing: [1, 0]))
        let y = try XCTUnwrap(Embedding(normalizing: [0, 1]))
        let negX = try XCTUnwrap(Embedding(normalizing: [-2, 0]))
        XCTAssertEqual(try XCTUnwrap(x.cosine(y)), 0, accuracy: 1e-6)
        XCTAssertEqual(try XCTUnwrap(x.cosine(negX)), -1, accuracy: 1e-6)
    }

    func testDimensionMismatchIsNilNotTrap() throws {
        let a = try XCTUnwrap(Embedding(normalizing: [1, 0]))
        let b = try XCTUnwrap(Embedding(normalizing: [1, 0, 0]))
        XCTAssertNil(a.cosine(b))
    }
}

final class HashedTrigramEmbedderTests: XCTestCase {

    func testRejectsDimensionOutsideRange() {
        XCTAssertThrowsError(try HashedTrigramEmbedder(dimension: 7))
        XCTAssertThrowsError(try HashedTrigramEmbedder(dimension: 4097))
        XCTAssertNoThrow(try HashedTrigramEmbedder(dimension: 8))
    }

    func testEmptyInputThrowsRatherThanReturningZeroVector() throws {
        let embedder = try Fixtures.embedder()
        XCTAssertThrowsError(try embedder.embedSynchronously("?!")) { error in
            XCTAssertEqual(error as? EmbeddingError, .emptyInput)
        }
    }

    func testDeterministicAndUnitLength() throws {
        let embedder = try Fixtures.embedder()
        let a = try embedder.embedSynchronously("Where is my order?")
        let b = try embedder.embedSynchronously("Where is my order?")
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.dimension, 256)
        let norm = a.values.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1, accuracy: 1e-4)
    }

    /// The property the whole cache relies on, asserted as an *ordering* rather
    /// than a magic number: every paraphrase in the corpus is closer to its own
    /// canonical prompt than any unrelated prompt is.
    func testParaphrasesAreCloserThanUnrelatedPrompts() throws {
        let embedder = try Fixtures.embedder()
        var unrelatedMax: Float = -1
        for topic in DemoCorpus.topics {
            let canonical = try embedder.embedSynchronously(topic.canonicalPrompt)
            for unrelated in DemoCorpus.unrelatedPrompts {
                let similarity = try XCTUnwrap(canonical.cosine(try embedder.embedSynchronously(unrelated)))
                unrelatedMax = max(unrelatedMax, similarity)
            }
        }
        var paraphraseMin: Float = 1
        for topic in DemoCorpus.topics {
            let canonical = try embedder.embedSynchronously(topic.canonicalPrompt)
            for paraphrase in topic.paraphrases {
                let similarity = try XCTUnwrap(canonical.cosine(try embedder.embedSynchronously(paraphrase)))
                paraphraseMin = min(paraphraseMin, similarity)
            }
        }
        XCTAssertGreaterThan(paraphraseMin, unrelatedMax,
                             "weakest paraphrase \(paraphraseMin) must beat strongest unrelated \(unrelatedMax)")
    }

    func testSameTopicParaphrasesBeatCrossTopicCanonicals() throws {
        let embedder = try Fixtures.embedder()
        let canonicals = try DemoCorpus.topics.map { try embedder.embedSynchronously($0.canonicalPrompt) }
        var violations: [String] = []
        for (index, topic) in DemoCorpus.topics.enumerated() {
            for paraphrase in topic.paraphrases {
                let query = try embedder.embedSynchronously(paraphrase)
                let own = try XCTUnwrap(canonicals[index].cosine(query))
                for (other, canonical) in canonicals.enumerated() where other != index {
                    let cross = try XCTUnwrap(canonical.cosine(query))
                    if cross >= own { violations.append("\(paraphrase) → \(DemoCorpus.topics[other].name)") }
                }
            }
        }
        XCTAssertTrue(violations.isEmpty, "paraphrases routed to the wrong topic: \(violations)")
    }
}

final class CachePolicyTests: XCTestCase {

    func testRejectsInvalidThreshold() {
        XCTAssertThrowsError(try CachePolicy(similarityThreshold: 0))
        XCTAssertThrowsError(try CachePolicy(similarityThreshold: 1.01))
        XCTAssertThrowsError(try CachePolicy(similarityThreshold: .nan))
        XCTAssertThrowsError(try CachePolicy(similarityThreshold: -0.5))
        XCTAssertNoThrow(try CachePolicy(similarityThreshold: 1))
    }

    func testRejectsInvalidBudgets() {
        XCTAssertThrowsError(try CachePolicy(maxEntries: 0))
        XCTAssertThrowsError(try CachePolicy(maxBytes: 0))
        XCTAssertThrowsError(try CachePolicy(falseHitTolerance: 1.5))
        XCTAssertThrowsError(try CachePolicy(falseHitTolerance: .nan))
    }

    func testShadowConfigurationRejectsInvalidRate() {
        XCTAssertThrowsError(try ShadowConfiguration(sampleRate: -0.1))
        XCTAssertThrowsError(try ShadowConfiguration(sampleRate: 1.1))
        XCTAssertThrowsError(try ShadowConfiguration(sampleRate: .nan))
        XCTAssertEqual(ShadowConfiguration.off.sampleRate, 0)
    }
}

final class WilsonIntervalTests: XCTestCase {

    /// k = 0, n = 12: the naive interval says "0% ± 0". Wilson says the upper
    /// bound is ~24%. Hand-computed: z² = 3.8416; centre = (3.8416/24)/(1 + 3.8416/12)
    /// = 0.12125; half-width = 1.96·sqrt(3.8416/576)/1.32013 = 0.12125.
    func testZeroSuccessesHasHonestUpperBound() throws {
        let interval = try XCTUnwrap(WilsonInterval(successes: 0, trials: 12))
        XCTAssertEqual(interval.estimate, 0)
        XCTAssertEqual(interval.lower, 0, accuracy: 1e-9)
        XCTAssertEqual(interval.upper, 0.2425, accuracy: 0.001)
    }

    func testHalfAndHalfIsSymmetric() throws {
        let interval = try XCTUnwrap(WilsonInterval(successes: 50, trials: 100))
        XCTAssertEqual(interval.estimate, 0.5)
        XCTAssertEqual(interval.lower, 0.4038, accuracy: 0.001)
        XCTAssertEqual(interval.upper, 0.5962, accuracy: 0.001)
    }

    func testDegenerateInputsReturnNil() {
        XCTAssertNil(WilsonInterval(successes: 0, trials: 0))
        XCTAssertNil(WilsonInterval(successes: 5, trials: 4))
        XCTAssertNil(WilsonInterval(successes: -1, trials: 4))
        XCTAssertNil(WilsonInterval(successes: 1, trials: 4, z: .nan))
    }
}
