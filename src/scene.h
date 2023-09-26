#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#include "tiny_gltf.h"

namespace tinygltf {
    class Model;
    class Node;
}

class Scene {
private:
    tinygltf::Model* model;
    int loadMaterial();
    int loadScene();
    void traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms);
    void loadNode(const tinygltf::Node& node);
    int loadGeom(const tinygltf::Node& node, const Geom::Transformation& transform);
    int loadCamera(const tinygltf::Node&, const glm::mat4& transform);
    const int height = 600;
public:
    Scene(std::string filename, int height);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
