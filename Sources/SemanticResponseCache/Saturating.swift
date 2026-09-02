/// Arithmetic that cannot trap.
///
/// Every counter and byte total in this package flows through here. The point is
/// not that a semantic cache is likely to overflow `Int` — it is that a cache is
/// exactly the kind of long-lived, always-on component where "unlikely" is
/// reached eventually, and a trap in a cache is a crash in the feature that
/// merely wanted an answer.
public enum Saturating {

    /// `a + b`, clamped to `Int.max` / `Int.min` instead of trapping.
    @inlinable
    public static func add(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.addingReportingOverflow(b)
        if overflow { return b > 0 ? Int.max : Int.min }
        return result
    }

    /// `a * b`, clamped instead of trapping.
    @inlinable
    public static func multiply(_ a: Int, _ b: Int) -> Int {
        let (result, overflow) = a.multipliedReportingOverflow(by: b)
        if overflow { return (a < 0) == (b < 0) ? Int.max : Int.min }
        return result
    }

    /// `Int(value)` without the trap.
    ///
    /// `Int(Double)` traps on NaN, on either infinity, and on any magnitude
    /// outside `Int`'s range. NaN maps to `fallback`; infinities and finite
    /// out-of-range values clamp to the nearest bound. The ceiling is derived from `Int.max` rather
    /// than a 64-bit literal because `Int` is 32 bits on watchOS.
    @inlinable
    public static func int(_ value: Double, fallback: Int = 0) -> Int {
        guard !value.isNaN else { return fallback }
        // `Double(Int.max)` rounds up to 2^63 on 64-bit, which is itself outside
        // `Int`, so compare with `>=` and clamp before converting.
        if value >= Double(Int.max) { return Int.max }
        if value <= Double(Int.min) { return Int.min }
        return Int(value)
    }

    /// Integer division that returns `fallback` for a zero divisor and for the
    /// one other trapping case, `Int.min / -1`.
    @inlinable
    public static func divide(_ a: Int, by b: Int, fallback: Int = 0) -> Int {
        if b == 0 { return fallback }
        if a == Int.min && b == -1 { return Int.max }
        return a / b
    }
}
