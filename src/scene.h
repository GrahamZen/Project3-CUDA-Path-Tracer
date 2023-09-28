#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.

namespace tinygltf {
    class Model;
    class Node;
}

class Scene {
private:
    static void loadExtensions(Material& material, const tinygltf::ExtensionMap& extensionMap);
    PbrMetallicRoughness loadPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbrMat);
    Camera& computeCameraParams(Camera& camera)const;
    int loadMaterial();
    bool loadTexture();
    int loadScene();
    void loadSettings();
    void traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms);
    void loadNode(const tinygltf::Node& node);
    int loadGeom(const tinygltf::Node& node, const Geom::Transformation& transform);
    bool loadCamera(const tinygltf::Node&, const glm::mat4& transform);
    TextureInfo crateTextureObj(int textureIndex, const tinygltf::Image& image);
    std::vector<cudaArray_t> dev_tex_arrs_vec;
    tinygltf::Model* model;
    const int defaultMatId = 0;
public:
    struct Settings
    {
        const std::string filename = "Settings.json";
        std::string envMapFilename;
        std::string gltfPath;
        Material defaultMat;
        RenderState defaultRenderState;
        bool readFromFile;
        bool antiAliasing;
    } settings;
    Scene(std::string filename);
    ~Scene();

    std::vector<cudaTextureObject_t> cuda_tex_vec;
    std::vector<TextureInfo> textures;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Camera> cameras;
    RenderState state;
};
