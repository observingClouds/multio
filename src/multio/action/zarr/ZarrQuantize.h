/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

/// Lossy quantization of the zarr action, applied by the compression workers to a field's float32 values just
/// before zstd (it only makes trailing mantissa bits zero; the stored data type stays float32 and any zarr reader
/// decodes it without knowing about the quantization).
///
/// Algorithms:
///   bitround (keep-bits: k, 1..23): round to nearest, ties to even, on the float32 mantissa, keeping k explicit
///       mantissa bits (Kloewer et al. 2021, as numcodecs BitRound / netCDF nc_def_var_quantize NC_QUANTIZE_BITROUND).
///       Relative error <= 2^-(k+1). A value whose rounding would carry into infinity is truncated instead.
///   absolute (absolute: e): round to the nearest multiple (ties to even) of D, the largest power of two <= e;
///       |error| <= D/2 <= e/2. Values with |x| >= 2^24 D are already multiples of D and stay unchanged.
///   range-bits (range-bits: n, 1..23): the absolute algorithm with D chosen per field (per level) as the largest
///       power of two <= (max - min) / 2^n of that field's finite values, i.e. at least GRIB/FA precision
///       "n bits over the field's range": |error| <= D/2 <= range / 2^(n+1). A constant field stays unchanged.
/// NaN and infinities pass unchanged; no finite value becomes infinite.
///
/// Metadata (CF-1.11 section 8.4 "Lossy compression via quantization"): the array gets `quantization` (name of a
/// container variable, a scalar int32 array without data in the store root) and, for bitround,
/// `quantization_nsb`; absolute: `quantization_step` and `quantization_maximum_absolute_error`; range-bits:
/// `quantization_range_bits`. The container has `algorithm` (bitround, or the non-CF names
/// absolute_power_of_two / range_bits_power_of_two) and `implementation`.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace multio::action::zarr {

enum class QuantMode
{
    None,       // lossless
    BitRound,   // keep-bits
    Absolute,   // fixed power-of-two step
    RangeBits,  // power-of-two step from the field's range
};

struct Quantization {
    QuantMode mode = QuantMode::None;
    int bits = 0;            // BitRound: kept mantissa bits; RangeBits: bits over the range
    double step = 0.0;       // Absolute: D (a power of two)
    double requested = 0.0;  // Absolute: the configured value (D = largest power of two <= requested)

    bool enabled() const { return mode != QuantMode::None; }

    /// Name of the CF container variable of this algorithm, e.g. quantization_bitround; empty without
    std::string containerName() const;

    /// Value of the container's `algorithm` attribute
    std::string algorithm() const;

    /// Members appended to an array's attributes object, starting with ','; empty without quantization
    std::string attributesJsonTail() const;

    /// Short description for logs, e.g. "bitround keep-bits 10"
    std::string describe() const;

    bool operator==(const Quantization& o) const { return mode == o.mode && bits == o.bits && step == o.step; }
};

/// Largest power of two <= x (x > 0, finite), as a double
double powerOfTwoFloor(double x);

/// Quantize n values in place (thread-safe, no state)
void quantize(const Quantization& q, float* values, std::size_t n);

/// One rule of the `quantize` configuration
struct QuantRule {
    std::vector<std::string> globs;  // fnmatch(3) patterns on the zarr array name (any one matches)
    std::string regexText;           // ECMAScript regex, whole name (std::regex_match); empty: none
    std::shared_ptr<const std::regex> regex;
    Quantization quantization;

    bool matches(const std::string& name) const;
};

/// Ordered rules (first match wins) and an optional default for names no rule matches
struct QuantRules {
    std::vector<QuantRule> rules;
    Quantization fallback;  // `default`; None: lossless

    bool empty() const { return rules.empty() && !fallback.enabled(); }

    /// Quantization of the array `name`
    Quantization lookup(const std::string& name) const;
};

}  // namespace multio::action::zarr
