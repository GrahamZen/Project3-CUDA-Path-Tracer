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
    tinygltf::Model* model;
    static PbrMetallicRoughness loadPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbrMat);
    static void loadExtensions(Material& material, const tinygltf::ExtensionMap& extensionMap);
    Camera& computeCameraParams(Camera& camera)const;
    int loadMaterial();
    int loadScene();
    void loadSettings();
    void traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms);
    void loadNode(const tinygltf::Node& node);
    int loadGeom(const tinygltf::Node& node, const Geom::Transformation& transform);
    bool loadCamera(const tinygltf::Node&, const glm::mat4& transform);
    const int defaultMatId = 0;
    const int height = 1000;
public:
    struct Settings
    {
        const std::string filename = "Settings.json";
        std::string gltfPath;
        Material defaultMat;
        RenderState defaultRenderState;
        bool readFromFile;
    } settings;
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Camera> cameras;
    RenderState state;
};
