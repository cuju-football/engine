// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's Matrix3, Matrix4, Plane and the AABB operations of its culling (base/math and
// base/geometry, Apache-2.0), in GRF's float order. Vectors and quaternions are cuju's ports.

#ifndef CUJU_RENDER_SCENE_GRF_MATH_H
#define CUJU_RENDER_SCENE_GRF_MATH_H

#include <cmath>
#include <vector>

#include "cuju/libm/tanf.h"
#include "cuju/quaternion.h"
#include "cuju/vector3.h"

namespace cuju::render::scene {
using cuju::Quaternion;
using cuju::Vector3;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEpsilon = 0.000001f;

struct Matrix3 {
    float e[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    Matrix3 operator*(const Matrix3 &m) const {
        Matrix3 r;
        for (int row = 0; row < 3; ++row)
            for (int c = 0; c < 3; ++c)
                r.e[row * 3 + c] = e[row * 3 + 0] * m.e[0 + c] + e[row * 3 + 1] * m.e[3 + c] +
                                   e[row * 3 + 2] * m.e[6 + c];
        return r;
    }
};

inline Matrix3 RotationMatrix(const Quaternion &q) {
    const float *el = q.elements;
    float xx = el[0] * el[0], xy = el[0] * el[1], xz = el[0] * el[2], xw = el[0] * el[3];
    float yy = el[1] * el[1], yz = el[1] * el[2], yw = el[1] * el[3];
    float zz = el[2] * el[2], zw = el[2] * el[3];
    Matrix3 r;
    r.e[0] = 1 - 2 * (yy + zz);
    r.e[1] = 2 * (xy - zw);
    r.e[2] = 2 * (xz + yw);
    r.e[3] = 2 * (xy + zw);
    r.e[4] = 1 - 2 * (xx + zz);
    r.e[5] = 2 * (yz - xw);
    r.e[6] = 2 * (xz - yw);
    r.e[7] = 2 * (yz + xw);
    r.e[8] = 1 - 2 * (xx + yy);
    return r;
}

inline Matrix3 ScaleMatrix(const Vector3 &s) {
    Matrix3 r;
    r.e[0] = s.coordinates[0];
    r.e[4] = s.coordinates[1];
    r.e[8] = s.coordinates[2];
    return r;
}

inline Quaternion QuaternionFromMatrix(const Matrix3 &m) {
    const float *t = m.e;
    Quaternion q;
    float n4;
    float tr = t[0] + t[4] + t[8];
    if (tr > 0.0f) {
        q.Set(t[7] - t[5], t[2] - t[6], t[3] - t[1], tr + 1.0f);
        n4 = q.elements[3];
    } else if ((t[0] > t[4]) && (t[0] > t[8])) {
        q.Set(1.0f + t[0] - t[4] - t[8], t[1] + t[3], t[2] + t[6], t[7] - t[5]);
        n4 = q.elements[0];
    } else if (t[4] > t[8]) {
        q.Set(t[1] + t[3], 1.0f + t[4] - t[0] - t[8], t[5] + t[7], t[2] - t[6]);
        n4 = q.elements[1];
    } else {
        q.Set(t[2] + t[6], t[5] + t[7], 1.0f + t[8] - t[0] - t[4], t[3] - t[1]);
        n4 = q.elements[2];
    }
    q.Scale(0.5f / std::sqrt(n4));
    return q;
}

inline Quaternion QuaternionFromDirection(const Vector3 &vec) {
    Vector3 z = vec;
    z.Normalize();
    Vector3 y(0, -1, 0);
    Vector3 x = y.GetCrossProduct(z);
    x.Normalize();
    y = z.GetCrossProduct(x);
    Matrix3 m;
    m.e[0] = x.coordinates[0];
    m.e[1] = x.coordinates[1];
    m.e[2] = x.coordinates[2];
    m.e[3] = y.coordinates[0];
    m.e[4] = y.coordinates[1];
    m.e[5] = y.coordinates[2];
    m.e[6] = z.coordinates[0];
    m.e[7] = z.coordinates[1];
    m.e[8] = z.coordinates[2];
    return QuaternionFromMatrix(m);
}

struct Matrix4 {
    float e[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    static Matrix4 Identity() {
        Matrix4 m;
        m.e[0] = m.e[5] = m.e[10] = m.e[15] = 1;
        return m;
    }
    void Assign3(const Matrix3 &m) {
        e[0] = m.e[0];
        e[1] = m.e[1];
        e[2] = m.e[2];
        e[4] = m.e[3];
        e[5] = m.e[4];
        e[6] = m.e[5];
        e[8] = m.e[6];
        e[9] = m.e[7];
        e[10] = m.e[8];
    }
    Matrix4 operator*(const Matrix4 &m) const {
        Matrix4 r;
        for (int row = 0; row < 4; ++row)
            for (int c = 0; c < 4; ++c)
                r.e[row * 4 + c] = e[row * 4 + 0] * m.e[0 + c] + e[row * 4 + 1] * m.e[4 + c] +
                                   e[row * 4 + 2] * m.e[8 + c] + e[row * 4 + 3] * m.e[12 + c];
        return r;
    }
    void SetTranslation(const Vector3 &t) {
        e[3] = t.coordinates[0];
        e[7] = t.coordinates[1];
        e[11] = t.coordinates[2];
    }
    Vector3 GetTranslation() const { return Vector3(e[3], e[7], e[11]); }
    void SetScale(const Vector3 &s) {
        e[0] = s.coordinates[0];
        e[5] = s.coordinates[1];
        e[10] = s.coordinates[2];
    }
    Matrix4 GetTransposed() const {
        Matrix4 m;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) m.e[r * 4 + c] = e[c * 4 + r];
        return m;
    }
    void Construct(const Vector3 &position, const Vector3 &scale, const Quaternion &rotation) {
        Assign3(RotationMatrix(rotation) * ScaleMatrix(scale));
        SetTranslation(position);
        e[12] = 0;
        e[13] = 0;
        e[14] = 0;
        e[15] = 1;
    }
    void ConstructInverse(const Vector3 &position, const Vector3 &scale,
                          const Quaternion &rotation) {
        Vector3 position_inverse = -position;
        Vector3 scale_inverse(1 / scale.coordinates[0], 1 / scale.coordinates[1],
                              1 / scale.coordinates[2]);
        Quaternion rotation_inverse = rotation.GetInverse();
        position_inverse *= scale_inverse;
        position_inverse = rotation_inverse * position_inverse;
        Assign3(ScaleMatrix(scale_inverse) * RotationMatrix(rotation_inverse));
        SetTranslation(position_inverse);
        e[12] = 0;
        e[13] = 0;
        e[14] = 0;
        e[15] = 1;
    }
    void ConstructProjection(float fov, float aspect, float z_near, float z_far) {
        float top = z_near * libm::tanf(fov * (kPi / 360.0f));
        float bottom = -top;
        float right = top * aspect;
        float left = -right;
        volatile float far_min_near = 0;
        if (std::fabs(z_far - z_near) > kEpsilon) far_min_near = 1.0f / (z_far - z_near);
        float fmn = far_min_near;
        e[0] = right != left ? (2 * z_near) / (right - left) : 0;
        e[1] = 0;
        e[2] = 0;
        e[3] = 0;
        e[4] = 0;
        e[5] = std::fabs(top - bottom) > kEpsilon ? (2 * z_near) / (top - bottom) : 0;
        e[6] = 0;
        e[7] = 0;
        e[8] = std::fabs(right - left) > kEpsilon ? (right + left) / (right - left) : 0;
        e[9] = std::fabs(top - bottom) > kEpsilon ? (top + bottom) / (top - bottom) : 0;
        e[10] = -(z_far + z_near) * fmn;
        e[11] = -1;
        e[12] = 0;
        e[13] = 0;
        e[14] = -(2 * z_far * z_near) * fmn;
        e[15] = 0;
        *this = GetTransposed();
    }
    void ConstructOrtho(float left, float right, float bottom, float top, float z_near,
                        float z_far) {
        e[0] = std::fabs(right - left) > kEpsilon ? 2 / (right - left) : 0;
        e[1] = 0;
        e[2] = 0;
        e[3] = 0;
        e[4] = 0;
        e[5] = std::fabs(top - bottom) > kEpsilon ? 2 / (top - bottom) : 0;
        e[6] = 0;
        e[7] = 0;
        e[8] = 0;
        e[9] = 0;
        e[10] = std::fabs(z_far - z_near) > kEpsilon ? -(2 / (z_far - z_near)) : 0;
        e[11] = 0;
        e[12] = std::fabs(right - left) > kEpsilon ? -((right + left) / (right - left)) : 0;
        e[13] = std::fabs(top - bottom) > kEpsilon ? -((top + bottom) / (top - bottom)) : 0;
        e[14] = std::fabs(z_far - z_near) > kEpsilon ? -((z_far + z_near) / (z_far - z_near)) : 0;
        e[15] = 1;
        *this = GetTransposed();
    }
    Matrix4 GetInverse() const;
    void MultiplyVec4(float x, float y, float z, float w, float &rx, float &ry, float &rz,
                      float &rw) const {
        rx = e[0] * x + e[1] * y + e[2] * z + e[3] * w;
        ry = e[4] * x + e[5] * y + e[6] * z + e[7] * w;
        rz = e[8] * x + e[9] * y + e[10] * z + e[11] * w;
        rw = e[12] * x + e[13] * y + e[14] * z + e[15] * w;
    }
};

// GRF's Plane: a point and a normal, determinant = -point . normal.
struct Plane {
    Vector3 point, normal;
    float Determinant() const { return -point.GetDotProduct(normal); }
};

struct Aabb {
    Vector3 min, max;
    void Reset() {
        const float min_v = 1.17549435e-38f / 100;
        const float max_v = 3.40282347e+38f / 100;
        min.Set(max_v, max_v, max_v);
        max.Set(min_v, min_v, min_v);
    }
    void Add(const Aabb &a) {
        for (int i = 0; i < 3; ++i) {
            if (min.coordinates[i] > a.min.coordinates[i])
                min.coordinates[i] = a.min.coordinates[i];
            if (max.coordinates[i] < a.max.coordinates[i])
                max.coordinates[i] = a.max.coordinates[i];
        }
    }
    Aabb Translated(const Vector3 &v) const {
        Aabb r = *this;
        r.min += v;
        r.max += v;
        return r;
    }
    Aabb Rotated(const Quaternion &rot) const;
    bool Intersects(const std::vector<Plane> &planes) const;
};

}  // namespace cuju::render::scene

#endif
