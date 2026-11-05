/*
 * (C) Copyright 2026- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation nor
 * does it submit to any jurisdiction.
 */

#include "ZarrQuantize.h"

#include <cmath>
#include <cstring>
#include <limits>

#include <fnmatch.h>

#include "eckit/exception/Exceptions.h"

#include "ZarrUtil.h"

namespace multio::action::zarr {

namespace {

static_assert(std::numeric_limits<float>::is_iec559, "IEEE float32 expected");

constexpr std::uint32_t EXPONENT = 0x7F800000u;

std::string jsonNumber(double d) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", d);
    return buf;
}

/// bitround of Kloewer et al. (2021) / numcodecs BitRound: round to nearest, ties to even, keep k mantissa bits
void bitRound(int keepBits, float* values, std::size_t n) {
    const unsigned drop = 23u - static_cast<unsigned>(keepBits);
    if (drop == 0) {
        return;
    }
    const std::uint32_t mask = ~((std::uint32_t{1} << drop) - 1u);
    const std::uint32_t half = (std::uint32_t{1} << (drop - 1u)) - 1u;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t u;
        std::memcpy(&u, values + i, sizeof(u));
        std::uint32_t r = (u + ((u >> drop) & 1u) + half) & mask;
        r = (r & EXPONENT) == EXPONENT ? (u & mask) : r;  // carry into infinity: truncate instead
        r = (u & EXPONENT) == EXPONENT ? u : r;           // NaN, infinity: unchanged
        std::memcpy(values + i, &r, sizeof(r));
    }
}

/// Round to the nearest multiple of the power of two `step` (normal float32), ties to even. Exact float32
/// arithmetic: x / step and back are exact scalings, and nearbyint rounds to an integer with ties to even in the
/// default rounding mode, keeping the sign of zero. (The (|y| + 2^23) - 2^23 trick is folded away by compilers
/// with value-unsafe floating-point optimisations, e.g. icpc's default -fp-model fast.)
void roundToStep(double step, float* values, std::size_t n) {
    const float d = static_cast<float>(step);
    const float inv = static_cast<float>(1.0 / step);
    const float limit = static_cast<float>(std::ldexp(step, 24));  // |x| >= 2^24 D: already a multiple of D
    for (std::size_t i = 0; i < n; ++i) {
        const float x = values[i];
        if (std::fabs(x) < limit) {  // false for NaN; true for every finite value if limit is infinite
            values[i] = std::nearbyint(x * inv) * d;
        }
    }
}

}  // namespace

//----------------------------------------------------------------------------------------------------------------------

double powerOfTwoFloor(double x) {
    int e = 0;
    std::frexp(x, &e);  // x = m 2^e, m in [0.5, 1)
    return std::ldexp(1.0, e - 1);
}

std::string Quantization::containerName() const {
    switch (mode) {
        case QuantMode::None:
            return std::string();
        case QuantMode::BitRound:
            return "quantization_bitround";
        case QuantMode::Absolute:
            return "quantization_absolute";
        case QuantMode::RangeBits:
            return "quantization_range_bits";
    }
    return std::string();
}

std::string Quantization::algorithm() const {
    switch (mode) {
        case QuantMode::None:
            return std::string();
        case QuantMode::BitRound:
            return "bitround";
        case QuantMode::Absolute:
            return "absolute_power_of_two";
        case QuantMode::RangeBits:
            return "range_bits_power_of_two";
    }
    return std::string();
}

std::string Quantization::attributesJsonTail() const {
    switch (mode) {
        case QuantMode::None:
            return std::string();
        case QuantMode::BitRound:
            return ",\"quantization\":" + jsonString(containerName()) + ",\"quantization_nsb\":" + std::to_string(bits);
        case QuantMode::Absolute:
            return ",\"quantization\":" + jsonString(containerName()) + ",\"quantization_step\":" + jsonNumber(step)
                 + ",\"quantization_maximum_absolute_error\":" + jsonNumber(step / 2);
        case QuantMode::RangeBits:
            return ",\"quantization\":" + jsonString(containerName())
                 + ",\"quantization_range_bits\":" + std::to_string(bits);
    }
    return std::string();
}

std::string Quantization::describe() const {
    switch (mode) {
        case QuantMode::None:
            return "lossless";
        case QuantMode::BitRound:
            return "bitround keep-bits " + std::to_string(bits);
        case QuantMode::Absolute:
            return "absolute step " + jsonNumber(step);
        case QuantMode::RangeBits:
            return "range-bits " + std::to_string(bits);
    }
    return "?";
}

void quantize(const Quantization& q, float* values, std::size_t n) {
    switch (q.mode) {
        case QuantMode::None:
            return;
        case QuantMode::BitRound:
            bitRound(q.bits, values, n);
            return;
        case QuantMode::Absolute:
            roundToStep(q.step, values, n);
            return;
        case QuantMode::RangeBits: {
            float lo = std::numeric_limits<float>::infinity();
            float hi = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < n; ++i) {
                const float x = values[i];
                if (std::isfinite(x)) {
                    lo = x < lo ? x : lo;
                    hi = x > hi ? x : hi;
                }
            }
            const double range = static_cast<double>(hi) - static_cast<double>(lo);
            if (!(range > 0) || !std::isfinite(range)) {
                return;  // constant (or no finite value): unchanged
            }
            const double step = powerOfTwoFloor(std::ldexp(range, -q.bits));
            if (step < static_cast<double>(std::numeric_limits<float>::min())) {
                return;  // finer than the normal float32 range: nothing to gain, left unchanged
            }
            roundToStep(step, values, n);
            return;
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------

bool QuantRule::matches(const std::string& domain, const std::string& name) const {
    if (!domains.empty()) {
        bool inDomain = false;
        for (const auto& d : domains) {
            inDomain = inDomain || ::fnmatch(d.c_str(), domain.c_str(), 0) == 0;
        }
        if (!inDomain) {
            return false;
        }
    }
    for (const auto& g : globs) {
        if (::fnmatch(g.c_str(), name.c_str(), 0) == 0) {
            return true;
        }
    }
    return regex && std::regex_match(name, *regex);
}

Quantization QuantRules::lookup(const std::string& domain, const std::string& name) const {
    for (const auto& r : rules) {
        if (r.matches(domain, name)) {
            return r.quantization;
        }
    }
    return fallback;
}

}  // namespace multio::action::zarr
