#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
#define TINYGLTF_IMPLEMENTATION
#include "scene.h"

template<typename T>
std::pair<const T*, int> getPrimitiveBuffer(tinygltf::Model* model, const tinygltf::Primitive& primitive, const std::string& type) {
    if (primitive.attributes.find(type) == primitive.attributes.end())return{ nullptr,0 };
    const tinygltf::Accessor& accessor = model->accessors[primitive.attributes.at(type)];
    const tinygltf::BufferView& bufferView = model->bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model->buffers[bufferView.buffer];
    const T* positions = reinterpret_cast<const T*>(&buffer.data[bufferView.byteOffset + accessor.byteOffset]);
    return { positions , accessor.count };
}

std::pair<const uint16_t*, int> getIndexBuffer(tinygltf::Model* model, const tinygltf::Primitive& primitive) {
    const tinygltf::Accessor& indexAccessor = model->accessors[primitive.indices];
    const tinygltf::BufferView& indexBufferView = model->bufferViews[indexAccessor.bufferView];
    const tinygltf::Buffer& indexBuffer = model->buffers[indexBufferView.buffer];
    const uint16_t* indices = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset]);
    return { indices , indexAccessor.count };
}

void updateTransform(const tinygltf::Node& node, std::vector<glm::mat4>& transforms) {
    glm::vec3 translation(0.0f);
    glm::quat rotation;
    glm::vec3 scale(1.0f);
    glm::mat4 transformation(1.0f);
    glm::mat4 t;
    if (!node.matrix.empty()) {
        transformation = glm::mat4(node.matrix[0], node.matrix[1], node.matrix[2], node.matrix[3],
            node.matrix[4], node.matrix[5], node.matrix[6], node.matrix[7],
            node.matrix[8], node.matrix[9], node.matrix[10], node.matrix[11],
            node.matrix[12], node.matrix[13], node.matrix[14], node.matrix[15]);
        t = transformation;
    }

    if (!node.translation.empty()) {
        translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        t = glm::translate(glm::mat4(1.0f), translation);
    }

    if (!node.rotation.empty()) {
        rotation = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
        t = glm::mat4_cast(rotation);
    }

    if (!node.scale.empty()) {
        scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        t = glm::scale(glm::mat4(1.0f), scale);
    }
    transforms.push_back(t);
}

Scene::Scene(std::string filename)
{
    std::cout << "Reading scene from " << filename << " ..." << std::endl;
    std::cout << " " << std::endl;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    model = new tinygltf::Model();
    bool ret = loader.LoadASCIIFromFile(model, &err, &warn, filename);
    if (!ret) {
        if (err.length() != 0)
            std::cerr << err;
        if (warn.length() != 0)
            std::cerr << warn;

        exit(-1);
    }
    loadScene();
    if (cameras.empty()) {
        genDefaultCamera();
    }
    auto& camera = cameras[0];
    state.camera = camera;
    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

}

Scene::~Scene()
{
    delete model;
}

Geom::Transformation evaluateTransform(std::vector<glm::mat4>& transforms) {
    Geom::Transformation t;
    t.transform = glm::mat4(1.0f);
    t.inverseTransform = glm::mat4(1.0f);
    t.invTranspose = glm::mat4(1.0f);
    for (auto it = transforms.rbegin(); it != transforms.rend(); ++it) {
        t.transform = t.transform * (*it);
    }
    t.inverseTransform = glm::inverse(t.transform);
    t.invTranspose = glm::transpose(t.inverseTransform);
    return t;
}

void Scene::traverseNode(const tinygltf::Node& node, std::vector<glm::mat4>& transforms) {
    if (node.camera != -1) {
        loadCamera(node, evaluateTransform(transforms).transform);
    }
    if (node.mesh != -1) {
        loadGeom(node, evaluateTransform(transforms));
    }
    updateTransform(node, transforms);
    for (int childIndex : node.children) {
        if (childIndex >= 0 && childIndex < model->nodes.size()) {
            const tinygltf::Node& childNode = model->nodes[childIndex];
            traverseNode(childNode, transforms);
            transforms.pop_back();
        }
    }
}

void Scene::loadNode(const tinygltf::Node& node)
{
    std::vector<glm::mat4> transforms;
    traverseNode(node, transforms);
}


int Scene::loadScene()
{
    RenderState& state = this->state;
    geoms.clear();
    int num_mat = loadMaterial();

    const tinygltf::Scene& scene = model->scenes[model->defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); i++)
    {
        const tinygltf::Node& node = model->nodes[scene.nodes[i]];
        loadNode(node);
    }
    if (num_mat == 0) {
        materials.emplace_back();
    }

    return 1;
}

int Scene::loadGeom(const tinygltf::Node& node, const Geom::Transformation& T)
{
    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model->meshes[node.mesh];

        for (size_t primitiveIndex = 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex) {
            const tinygltf::Primitive& primitive = mesh.primitives[primitiveIndex];

            if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
                int materialId = materials.empty() ? defaultMatId : primitive.material;

                auto [positions, posCnt] = getPrimitiveBuffer<float>(model, primitive, "POSITION");
                auto [normals, norCnt] = getPrimitiveBuffer<float>(model, primitive, "NORMAL");
                auto [uvs, uvCnt] = getPrimitiveBuffer<float>(model, primitive, "TEXCOORD_0");
                auto [indices, indCnt] = getIndexBuffer(model, primitive);

                for (size_t i = 0; i < indCnt; i += 3) {
                    const size_t v0Id = indices[i];
                    const size_t v1Id = indices[i + 1];
                    const size_t v2Id = indices[i + 2];
                    Geom triangle;
                    triangle.materialid = materialId;
                    triangle.v0 = glm::vec3(positions[v0Id * 3], positions[v0Id * 3 + 1], positions[v0Id * 3 + 2]);
                    triangle.v1 = glm::vec3(positions[v1Id * 3], positions[v1Id * 3 + 1], positions[v1Id * 3 + 2]);
                    triangle.v2 = glm::vec3(positions[v2Id * 3], positions[v2Id * 3 + 1], positions[v2Id * 3 + 2]);
                    if (normals) {
                        triangle.normal0 = glm::vec3(normals[v0Id * 3], normals[v0Id * 3 + 1], normals[v0Id * 3 + 2]);
                        triangle.normal1 = glm::vec3(normals[v1Id * 3], normals[v1Id * 3 + 1], normals[v1Id * 3 + 2]);
                        triangle.normal2 = glm::vec3(normals[v2Id * 3], normals[v2Id * 3 + 1], normals[v2Id * 3 + 2]);
                    }
                    if (uvs) {
                        triangle.uv0 = glm::vec2(normals[v0Id * 2], normals[v0Id * 2 + 1]);
                        triangle.uv1 = glm::vec2(normals[v1Id * 2], normals[v1Id * 2 + 1]);
                        triangle.uv2 = glm::vec2(normals[v2Id * 2], normals[v2Id * 2 + 1]);
                    }
                    geoms.push_back(triangle);
                }
            }
        }
    }
    return geoms.size();
}

void Scene::genDefaultCamera()
{
    std::cout << "Generating default Camera ..." << std::endl;
    Camera camera;
    camera.resolution = glm::vec2(height, height);
    float yscaled = tan(camera.fov.y * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);
    cameras.push_back(camera);
}

int Scene::loadCamera(const tinygltf::Node& node, const glm::mat4& transform)
{
    std::cout << "Loading Camera ..." << std::endl;
    Camera camera;
    camera.resolution.y = height;
    float fovy;
    const tinygltf::Camera& gltfCamera = model->cameras[node.camera];
    if (node.translation.size() == 3)
        camera.position = glm::vec3(transform * glm::vec4(node.translation[0], node.translation[1], node.translation[2], 1.0f));
    if (node.rotation.size() == 4)
    {
        glm::mat4 rot = transform * glm::mat4_cast(glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]));
        camera.view = glm::vec3(rot * glm::vec4(camera.view, 0.0f));
        camera.up = glm::vec3(rot * glm::vec4(camera.up, 0.0f));
    }
    if (gltfCamera.type == "perspective")
    {
        const tinygltf::PerspectiveCamera& perspective = gltfCamera.perspective;
        fovy = glm::degrees(perspective.yfov);
        camera.resolution.x = perspective.aspectRatio * camera.resolution.y;
    }
    else if (gltfCamera.type == "orthographic")
    {
        const tinygltf::OrthographicCamera& ortho = gltfCamera.orthographic;
        std::cout << "Orthographic Camera not implemented." << std::endl;
        return 0;
    }
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);
    cameras.push_back(camera);
}


PbrMetallicRoughness&& Scene::loadPbrMetallicRoughness(const tinygltf::PbrMetallicRoughness& pbrMat)
{
    PbrMetallicRoughness result;

    // Load base color factor
    for (size_t i = 0; i < pbrMat.baseColorFactor.size(); ++i) {
        result.baseColorFactor[i] = pbrMat.baseColorFactor[i];
    }

    // Load base color texture
    result.baseColorTexture.index = pbrMat.baseColorTexture.index;

    // Load metallic factor
    result.metallicFactor = pbrMat.metallicFactor;

    // Load roughness factor
    result.roughnessFactor = pbrMat.roughnessFactor;

    // Load metallic roughness texture
    result.metallicRoughnessTexture.index = pbrMat.metallicRoughnessTexture.index;

    return std::move(result);
}

void Scene::loadExtensions(Material& material, const tinygltf::ExtensionMap& extensionMap)
{
    for (const auto& entry : extensionMap) {
        const std::string& extensionName = entry.first;
        const auto& extensionValue = entry.second;

        if (extensionName == "KHR_materials_ior") {
            material.dielectric.eta = extensionValue.Get("ior").Get<double>();
        }
        else if (extensionName == "KHR_materials_specular") {
            // Extract specular color factor for metallic material
            material.type = Material::Type::DIELECTRIC;
            material.dielectric.eta = extensionValue.Get("specularFactor").Get<double>();
        }
        else if (extensionName == "KHR_materials_transmission") {
            // Extract transmission factor for dielectric material
        }
        else {
            std::cerr << extensionName << " not supported." << std::endl;
        }
        // Add more conditions for other extensions as needed
    }
}


int Scene::loadMaterial() {
    const std::vector<tinygltf::Material>& gltfMaterials = model->materials;
    materials.clear(); // Clear existing materials

    for (size_t i = 0; i < gltfMaterials.size(); ++i) {
        const tinygltf::Material& gltfMaterial = gltfMaterials[i];
        Material material;
        material.pbrMetallicRoughness = loadPbrMetallicRoughness(gltfMaterial.pbrMetallicRoughness);

        const auto& emissiveFactor = gltfMaterial.emissiveFactor;
        material.emissiveFactor = glm::vec3(emissiveFactor[0], emissiveFactor[1], emissiveFactor[2]);
        material.alphaCutoff = gltfMaterial.alphaCutoff;
        material.doubleSided = gltfMaterial.doubleSided;

        if (gltfMaterial.normalTexture.index >= 0)
            material.normalTexture.index = gltfMaterial.normalTexture.index;
        if (gltfMaterial.occlusionTexture.index >= 0)
            material.occlusionTexture.index = gltfMaterial.occlusionTexture.index;
        if (gltfMaterial.emissiveTexture.index >= 0)
            material.emissiveTexture.index = gltfMaterial.emissiveTexture.index;

        if (gltfMaterial.extensions.size() != 0)
            loadExtensions(material, gltfMaterial.extensions);

        // Add the material to the vector
        materials.push_back(material);
    }

    return static_cast<int>(materials.size());
}
