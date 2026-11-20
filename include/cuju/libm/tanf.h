// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================
//
// Copyright (C) 2018-2024 Free Software Foundation, Inc.
//
//   This library is free software; you can redistribute it and/or
//   modify it under the terms of the GNU Lesser General Public
//   License as published by the Free Software Foundation; either
//   version 2.1 of the License, or (at your option) any later version.
//
//   This library is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//   Lesser General Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public
//   License along with this library; if not, see
//   <https://www.gnu.org/licenses/>.
//
// Ported to C++ for host and device from glibc 2.39 sysdeps/ieee754/flt-32/s_tanf.c, k_tanf.c and
// s_sincosf.h by the cuju Authors, 2026.
// SPDX-License-Identifier: LGPL-2.1-or-later AND SunPro

// Float tanf of glibc 2.39, bit for bit on host and device. x86-64 has a single build of it, for
// the base instruction set, so nothing fuses.
// Exact: float arithmetic in the kernel, double arithmetic in the reduction, no contraction.
#pragma once

#include "../host_device.h"
#include "common.h"
#include "sincosf.h"

namespace cuju {
namespace libm {

// s_sincosf.h's `reduce_fast` as glibc's build without FMA rounds it, with the product and the
// subtraction apart.
CUJU_HOST_DEVICE double ReduceSmallUnfused(double argument, const SinCosFloatConstants *constants,
                                           int *quadrant) {
    double scaled = argument * constants->inverseHalfPiScaled;
    int rounded = ((int32_t)scaled + 0x800000) >> 24;
    *quadrant = rounded;
    return argument - (double)rounded * constants->halfPi;
}

// k_tanf.c's `__kernel_tanf`, the tangent of the sum of head and tail for a direction of 1 and
// minus its reciprocal for a direction of -1, for a sum up to about a quarter of pi in magnitude.
CUJU_HOST_DEVICE float TanFloatKernel(float head, float tail, int direction) {
    const float quarterPi = 0x1.921fb4p-1f, quarterPiLow = 0x1.4442d0p-25f;
    const float coefficient0 = 0x1.555556p-2f, coefficient1 = 0x1.111112p-3f,
                coefficient2 = 0x1.ba1ba2p-5f, coefficient3 = 0x1.664f48p-6f,
                coefficient4 = 0x1.226e3ep-7f, coefficient5 = 0x1.d6d22cp-9f,
                coefficient6 = 0x1.7dbc90p-10f, coefficient7 = 0x1.344d90p-11f,
                coefficient8 = 0x1.026f72p-12f, coefficient9 = 0x1.47e88ap-14f,
                coefficient10 = 0x1.2b80f4p-14f, coefficient11 = -0x1.375cbep-16f,
                coefficient12 = 0x1.b2a708p-16f;
    float square, fourthPower, odd, even, cube, sum;
    int32_t word = FloatWord(head), magnitudeWord = word & 0x7fffffff;
    if (magnitudeWord < 0x39000000) {
        if ((magnitudeWord | (direction + 1)) == 0) return 1.0f / fabsf(head);
        if (direction == 1) return head;
        return -1.0f / head;
    }
    if (magnitudeWord >= 0x3f2ca140) {
        if (word < 0) {
            head = -head;
            tail = -tail;
        }
        head = (quarterPi - head) + (quarterPiLow - tail);
        tail = 0.0f;
        if (fabsf(head) < 0x1p-13f)
            return (float)((1 - ((word >> 30) & 2)) * direction) *
                   (1.0f - (float)(2 * direction) * head);
    }
    square = head * head;
    fourthPower = square * square;
    odd = coefficient1 +
          fourthPower *
              (coefficient3 +
               fourthPower *
                   (coefficient5 +
                    fourthPower * (coefficient7 +
                                   fourthPower * (coefficient9 + fourthPower * coefficient11))));
    even = square *
           (coefficient2 +
            fourthPower *
                (coefficient4 +
                 fourthPower *
                     (coefficient6 +
                      fourthPower * (coefficient8 + fourthPower * (coefficient10 +
                                                                   fourthPower * coefficient12)))));
    cube = square * head;
    odd = tail + square * (cube * (odd + even) + tail);
    odd += coefficient0 * cube;
    sum = head + odd;
    if (magnitudeWord >= 0x3f2ca140) {
        float unit = (float)direction;
        return (float)(1 - ((word >> 30) & 2)) *
               (unit - 2.0f * (head - (sum * sum / (sum + unit) - odd)));
    }
    if (direction == 1) return sum;
    // The reciprocal from the high 12 significand bits of the sum and of the quotient.
    float sumHigh = FloatFromWord(FloatWord(sum) & 0xfffff000);
    float sumLow = odd - (sumHigh - head);
    float quotient = -1.0f / sum;
    float quotientHigh = FloatFromWord(FloatWord(quotient) & 0xfffff000);
    float error = 1.0f + quotientHigh * sumHigh;
    return quotientHigh + quotient * (error + quotientHigh * sumLow);
}

CUJU_HOST_DEVICE_NO_INLINE float tanf(float argument) {
    int32_t magnitudeWord = FloatWord(argument) & 0x7fffffff;
    if (magnitudeWord <= 0x3f490fda) return TanFloatKernel(argument, 0.0f, 1);
    if (magnitudeWord >= 0x7f800000) return argument - argument;
    double reduced;
    int quadrant;
    if (MagnitudeTopBits(argument) < MagnitudeTopBits(120.0f)) {
        reduced = ReduceSmallUnfused(argument, &CUJU_TABLE(SIN_COS_FLOAT_CONSTANTS)[0], &quadrant);
    } else {
        uint32_t bits = FloatBits(argument);
        reduced = ReduceLarge(bits, &quadrant);
        if (bits >> 31) reduced = -reduced;
    }
    float head = (float)reduced;
    float tail = (float)(reduced - (double)head);
    return TanFloatKernel(head, tail, 1 - ((quadrant & 1) << 1));
}

}  // namespace libm
}  // namespace cuju
