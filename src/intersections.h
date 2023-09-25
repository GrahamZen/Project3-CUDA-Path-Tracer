#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ float triangleIntersectionTest(Geom triangle, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv)
{
    glm::vec3 v0 = triangle.v0;
    glm::vec3 v1 = triangle.v1;
    glm::vec3 v2 = triangle.v2;
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 tvec = r.origin - v0;
    glm::vec3 pvec = glm::cross(r.direction, v0v2);
    float det = dot(v0v1, pvec);
    if (fabs(det) < 1e-5)
        return -1;
    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float invDet = 1.0 / det;
    float t = dot(qvec, v0v2) * invDet;
    float t_min = -1e38f;
    float t_max = 1e38f;
    if (t >= t_max || t <= t_min)
        return -1;
    float u = dot(tvec, pvec) * invDet;
    float v = dot(r.direction, qvec) * invDet;
    if (v < 0 || u + v > 1 || u < 0 || u > 1)
        return -1;
    float w = 1 - u - v;
    intersectionPoint = w * v0 + u * v1 + v * v2;
    uv = w * triangle.uv0 + u * triangle.uv1 + v * triangle.uv2;
    normal = glm::vec3(w * triangle.normal0 + u * triangle.normal1 + v * triangle.normal2);
    if (glm::dot(r.direction, normal) >= 0)
        normal = -normal;
    return t;
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}