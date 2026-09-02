// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "semantic-response-cache-kit",
    // Only platforms CI actually builds are declared. Linux needs no declaration;
    // the demo app's CI builds for `generic/platform=iOS Simulator`.
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(name: "SemanticResponseCache", targets: ["SemanticResponseCache"]),
        .library(name: "SemanticResponseCacheUI", targets: ["SemanticResponseCacheUI"])
    ],
    targets: [
        .target(name: "SemanticResponseCache"),
        .target(name: "SemanticResponseCacheUI", dependencies: ["SemanticResponseCache"]),
        .testTarget(name: "SemanticResponseCacheTests", dependencies: ["SemanticResponseCache"])
    ]
)
