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
__host__ __device__ glm::vec3 multiplyMV(const glm::mat4& m, const glm::vec4& v) {
    return glm::vec3(m * v);
}


/*
* return val:
* x: t
* y: v0's weight
* z: v1's weight
*/
__host__ __device__ float3 triangleIntersectionTest(TriangleDetail triangle, Ray r)
{
    glm::vec3 v0 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v0, 1.f));
    glm::vec3 v1 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v1, 1.f));
    glm::vec3 v2 = multiplyMV(triangle.t.transform, glm::vec4(triangle.v2, 1.f));
    glm::vec3 v0v1 = v1 - v0;
    glm::vec3 v0v2 = v2 - v0;
    glm::vec3 tvec = r.origin - v0;
    glm::vec3 pvec = glm::cross(r.direction, v0v2);
    float det = dot(v0v1, pvec);
    if (fabs(det) < 1e-5)
        return { -1,0,0 };
    glm::vec3 qvec = glm::cross(tvec, v0v1);
    float invDet = 1.0 / det;
    float t = dot(qvec, v0v2) * invDet;
    float t_min = -1e38f;
    float t_max = 1e38f;
    if (t >= t_max || t <= t_min)
        return { -1,0,0 };
    float u = dot(tvec, pvec) * invDet;
    float v = dot(r.direction, qvec) * invDet;
    if (v < 0 || u + v > 1 || u < 0 || u > 1)
        return { -1,0,0 };
    return { t, 1 - u - v, u };
}

__device__ bool intersectTBB(const Ray& ray, const TBB& tbb, float& tmin) {
    float tmax, tymin, tymax, tzmin, tzmax;
    tmin = (tbb.min.x - ray.origin.x) / ray.direction.x;
    tmax = (tbb.max.x - ray.origin.x) / ray.direction.x;
    tymin = (tbb.min.y - ray.origin.y) / ray.direction.y;
    tymax = (tbb.max.y - ray.origin.y) / ray.direction.y;
    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;
    tzmin = (tbb.min.z - ray.origin.z) / ray.direction.z;
    tzmax = (tbb.max.z - ray.origin.z) / ray.direction.z;
    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    if (tzmin > tmin)
        tmin = tzmin;
    if (tzmax < tmax)
        tmax = tzmax;
    return true;
}

__device__ int mapDirToIdx(const glm::vec3& r) {
    return 0;
}


__device__ float3 sceneIntersectionTest(TriangleDetail* triangles, TBVHNode* nodes, int nodesNum, Ray r, int& hitIdx) {
    int tbbOffset = mapDirToIdx(r.direction);
    int currentNodeIdx = 0;
    float t_min = FLT_MAX;
    float3 tmp_t;
    float3 t;
    hitIdx = -1;
    int tmpHitIdx = -1;
    while (currentNodeIdx != -1 && currentNodeIdx < nodesNum)
    {
        float tbbTmin = FLT_MAX;
        const TBVHNode& currNode = nodes[tbbOffset * nodesNum + currentNodeIdx];
        if (intersectTBB(r, currNode.tbb, tbbTmin) && tbbTmin < t_min) {
            if (currNode.isLeaf)
            {
                t = triangleIntersectionTest(triangles[currNode.triId], r);
                if (t.x > 0.0f && t_min > t.x) {
                    tmp_t = t;
                    t_min = t.x;
                    hitIdx = currNode.triId;
                }
            }
            currentNodeIdx++;
        }
        else
        {
            currentNodeIdx = currNode.miss;
        }
    }
    return tmp_t;
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}