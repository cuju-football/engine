// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "grf_math.h"

namespace cuju::render::scene {
Matrix4 Matrix4::GetInverse() const {
    float tmp[12];
    float src[16];
    float dst[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        src[i] = e[i * 4];
        src[i + 4] = e[i * 4 + 1];
        src[i + 8] = e[i * 4 + 2];
        src[i + 12] = e[i * 4 + 3];
    }
    tmp[0] = src[10] * src[15];
    tmp[1] = src[11] * src[14];
    tmp[2] = src[9] * src[15];
    tmp[3] = src[11] * src[13];
    tmp[4] = src[9] * src[14];
    tmp[5] = src[10] * src[13];
    tmp[6] = src[8] * src[15];
    tmp[7] = src[11] * src[12];
    tmp[8] = src[8] * src[14];
    tmp[9] = src[10] * src[12];
    tmp[10] = src[8] * src[13];
    tmp[11] = src[9] * src[12];
    dst[0] = tmp[0] * src[5] + tmp[3] * src[6] + tmp[4] * src[7];
    dst[0] -= tmp[1] * src[5] + tmp[2] * src[6] + tmp[5] * src[7];
    dst[1] = tmp[1] * src[4] + tmp[6] * src[6] + tmp[9] * src[7];
    dst[1] -= tmp[0] * src[4] + tmp[7] * src[6] + tmp[8] * src[7];
    dst[2] = tmp[2] * src[4] + tmp[7] * src[5] + tmp[10] * src[7];
    dst[2] -= tmp[3] * src[4] + tmp[6] * src[5] + tmp[11] * src[7];
    dst[3] = tmp[5] * src[4] + tmp[8] * src[5] + tmp[11] * src[6];
    dst[3] -= tmp[4] * src[4] + tmp[9] * src[5] + tmp[10] * src[6];
    dst[4] = tmp[1] * src[1] + tmp[2] * src[2] + tmp[5] * src[3];
    dst[4] -= tmp[0] * src[1] + tmp[3] * src[2] + tmp[4] * src[3];
    dst[5] = tmp[0] * src[0] + tmp[7] * src[2] + tmp[8] * src[3];
    dst[5] -= tmp[1] * src[0] + tmp[6] * src[2] + tmp[9] * src[3];
    dst[6] = tmp[3] * src[0] + tmp[6] * src[1] + tmp[11] * src[3];
    dst[6] -= tmp[2] * src[0] + tmp[7] * src[1] + tmp[10] * src[3];
    dst[7] = tmp[4] * src[0] + tmp[9] * src[1] + tmp[10] * src[2];
    dst[7] -= tmp[5] * src[0] + tmp[8] * src[1] + tmp[11] * src[2];
    tmp[0] = src[2] * src[7];
    tmp[1] = src[3] * src[6];
    tmp[2] = src[1] * src[7];
    tmp[3] = src[3] * src[5];
    tmp[4] = src[1] * src[6];
    tmp[5] = src[2] * src[5];
    tmp[6] = src[0] * src[7];
    tmp[7] = src[3] * src[4];
    tmp[8] = src[0] * src[6];
    tmp[9] = src[2] * src[4];
    tmp[10] = src[0] * src[5];
    tmp[11] = src[1] * src[4];
    dst[8] = tmp[0] * src[13] + tmp[3] * src[14] + tmp[4] * src[15];
    dst[8] -= tmp[1] * src[13] + tmp[2] * src[14] + tmp[5] * src[15];
    dst[9] = tmp[1] * src[12] + tmp[6] * src[14] + tmp[9] * src[15];
    dst[9] -= tmp[0] * src[12] + tmp[7] * src[14] + tmp[8] * src[15];
    dst[10] = tmp[2] * src[12] + tmp[7] * src[13] + tmp[10] * src[15];
    dst[10] -= tmp[3] * src[12] + tmp[6] * src[13] + tmp[11] * src[15];
    dst[11] = tmp[5] * src[12] + tmp[8] * src[13] + tmp[11] * src[14];
    dst[11] -= tmp[4] * src[12] + tmp[9] * src[13] + tmp[10] * src[14];
    dst[12] = tmp[2] * src[10] + tmp[5] * src[11] + tmp[1] * src[9];
    dst[12] -= tmp[4] * src[11] + tmp[0] * src[9] + tmp[3] * src[10];
    dst[13] = tmp[8] * src[11] + tmp[0] * src[8] + tmp[7] * src[10];
    dst[13] -= tmp[6] * src[10] + tmp[9] * src[11] + tmp[1] * src[8];
    dst[14] = tmp[6] * src[9] + tmp[11] * src[11] + tmp[3] * src[8];
    dst[14] -= tmp[10] * src[11] + tmp[2] * src[8] + tmp[7] * src[9];
    dst[15] = tmp[10] * src[10] + tmp[4] * src[8] + tmp[9] * src[9];
    dst[15] -= tmp[8] * src[9] + tmp[11] * src[10] + tmp[5] * src[8];
    float det = src[0] * dst[0] + src[1] * dst[1] + src[2] * dst[2] + src[3] * dst[3];
    if (det != 0.0f) {
        det = 1 / det;
        for (int j = 0; j < 16; j++) dst[j] *= det;
    }
    Matrix4 r;
    for (int i = 0; i < 16; ++i) r.e[i] = dst[i];
    return r;
}

Aabb Aabb::Rotated(const Quaternion &rot) const {
    Vector3 v[8] = {
        Vector3(min.coordinates[0], min.coordinates[1], min.coordinates[2]),
        Vector3(max.coordinates[0], min.coordinates[1], min.coordinates[2]),
        Vector3(min.coordinates[0], max.coordinates[1], min.coordinates[2]),
        Vector3(max.coordinates[0], max.coordinates[1], min.coordinates[2]),
        Vector3(min.coordinates[0], min.coordinates[1], max.coordinates[2]),
        Vector3(max.coordinates[0], min.coordinates[1], max.coordinates[2]),
        Vector3(min.coordinates[0], max.coordinates[1], max.coordinates[2]),
        Vector3(max.coordinates[0], max.coordinates[1], max.coordinates[2]),
    };
    for (Vector3 &p : v) p = rot * p;
    Aabb r;
    r.Reset();
    for (const Vector3 &p : v) {
        for (int i = 0; i < 3; ++i) {
            if (p.coordinates[i] < r.min.coordinates[i]) r.min.coordinates[i] = p.coordinates[i];
            if (p.coordinates[i] > r.max.coordinates[i]) r.max.coordinates[i] = p.coordinates[i];
        }
    }
    return r;
}

bool Aabb::Intersects(const std::vector<Plane> &planes) const {
    Vector3 pts[8] = {
        Vector3(min.coordinates[0], min.coordinates[1], min.coordinates[2]),
        Vector3(min.coordinates[0], max.coordinates[1], min.coordinates[2]),
        Vector3(min.coordinates[0], max.coordinates[1], max.coordinates[2]),
        Vector3(min.coordinates[0], min.coordinates[1], max.coordinates[2]),
        Vector3(max.coordinates[0], min.coordinates[1], min.coordinates[2]),
        Vector3(max.coordinates[0], max.coordinates[1], min.coordinates[2]),
        Vector3(max.coordinates[0], max.coordinates[1], max.coordinates[2]),
        Vector3(max.coordinates[0], min.coordinates[1], max.coordinates[2]),
    };
    bool inside = true;
    for (const Plane &p : planes) {
        inside = false;
        float det = p.Determinant();
        for (const Vector3 &a : pts) {
            float d = det + p.normal.coordinates[0] * a.coordinates[0] +
                      p.normal.coordinates[1] * a.coordinates[1] +
                      p.normal.coordinates[2] * a.coordinates[2];
            if (d > 0) {
                inside = true;
                break;
            }
        }
        if (!inside) break;
    }
    return inside;
}

}
