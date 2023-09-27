#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"
#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum BsdfSampleType
{
    DIFFUSE_REFL = 1,
    SPEC_REFL = 1 << 1,
    SPEC_TRANS = 1 << 2,
    MICROFACET_REFL = 1 << 3,
    MICROFACET_TRANS = 1 << 4,
    PLASTIC = 1 << 5,
    DIFFUSE_TRANS = 1 << 6
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

__device__ static Ray SpawnRay(glm::vec3 pos, glm::vec3 wi) {
    return Ray{ pos + wi * 0.01f, wi };
}

struct Geom
{
    struct Transformation {
        glm::mat4 transform;
        glm::mat4 inverseTransform;
        glm::mat4 invTranspose;
    }t;
    int materialid;
    glm::vec3 v0, v1, v2;
    glm::vec3 normal0, normal1, normal2;
    glm::vec2 uv0, uv1, uv2;
};

struct TextureInfo {
    int index;
};

struct NormalTextureInfo {
    int index{ -1 };    // required
    int texCoord{ 0 };  // The set index of texture's TEXCOORD attribute used for
    // texture coordinate mapping.
    double scale{
        1.0 };  // scaledNormal = normalize((<sampled normal texture value>
    // * 2.0 - 1.0) * vec3(<normal scale>, <normal scale>, 1.0))


    NormalTextureInfo() = default;
    DEFAULT_METHODS(NormalTextureInfo)
        bool operator==(const NormalTextureInfo&) const;
};


struct OcclusionTextureInfo {
    int index{ -1 };    // required
    int texCoord{ 0 };  // The set index of texture's TEXCOORD attribute used for
    // texture coordinate mapping.
    double strength{ 1.0 };  // occludedColor = lerp(color, color * <sampled
    // occlusion texture value>, <occlusion strength>)

    // Filled when SetStoreOriginalJSONForExtrasAndExtensions is enabled.

    OcclusionTextureInfo() = default;
    DEFAULT_METHODS(OcclusionTextureInfo)
        bool operator==(const OcclusionTextureInfo&) const;
};

struct PbrMetallicRoughness {
    glm::vec4 baseColorFactor;  // len = 4. default [1,1,1,1]
    TextureInfo baseColorTexture;
    double metallicFactor{ 1.0 };   // default 1
    double roughnessFactor{ 1.0 };  // default 1
    TextureInfo metallicRoughnessTexture;

    PbrMetallicRoughness()
        : baseColorFactor(glm::vec4{ 1. }) {}
    DEFAULT_METHODS(PbrMetallicRoughness)
        bool operator==(const PbrMetallicRoughness&) const;
};

struct Material {
    enum Type {
        UNKNOWN = 0,
        DIFFUSE = 1,
        DIELECTRIC = 1 << 1,
        METAL = 1 << 2,
        ROUGH_DIELECTRIC = 1 << 3,
        PLASTIC = 1 << 4
    };
    uint32_t type = Type::DIFFUSE;

    glm::vec3 emissiveFactor = glm::vec3(1.f);  // length 3. default [0, 0, 0]
    // std::string alphaMode;               // default "OPAQUE"
    double alphaCutoff{ 0.5 };             // default 0.5
    bool doubleSided{ false };             // default false;

    PbrMetallicRoughness pbrMetallicRoughness;

    NormalTextureInfo normalTexture;
    OcclusionTextureInfo occlusionTexture;
    TextureInfo emissiveTexture;

    struct Dielectric {
        float eta = 0.f;
        glm::vec3 specularColorFactor = glm::vec3(0.f);
    }dielectric;

    struct Metal {
        glm::vec3 etat = glm::vec3(0.f);
        glm::vec3 k = glm::vec3(0.f);
    }metal;

    __host__ __device__ Material() : dielectric(), metal() {}
    __host__ __device__ ~Material() = default;
    __host__ __device__ Material(const Material&) = default;
    __host__ __device__ Material(Material&&) TINYGLTF_NOEXCEPT = default;
    __host__ __device__ Material& operator=(const Material&) = default;
    __host__ __device__ Material& operator=(Material&&) TINYGLTF_NOEXCEPT = default;
    __host__ __device__ bool operator==(const Material&) const;
};



struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    bool isCached = false;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    glm::vec3 throughput;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    glm::vec3 pos;
    float t;
    glm::vec3 surfaceNormal;
    int materialId;
    glm::vec3 woW;
    glm::vec2 uv;
};
