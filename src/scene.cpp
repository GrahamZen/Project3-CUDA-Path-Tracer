#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <iostream>
// #define TINYGLTF_NOEXCEPTION // optional. disable exception handling.
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

template<typename T>
std::pair<const T*, int> getPrimitiveBuffer(tinygltf::Model* model, const tinygltf::Primitive& primitive, const string& type) {
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

Scene::Scene(string filename, int height = 1600)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
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
}

Scene::~Scene()
{
    delete model;
}

Geom::Transformation evaluateTransform(std::vector<glm::mat4>& transforms) {
    Geom::Transformation T;
    T.transform = glm::mat4(1.0f);
    T.inverseTransform = glm::mat4(1.0f);
    T.invTranspose = glm::mat4(1.0f);
    for (auto it = transforms.rbegin(); it != transforms.rend(); ++it) {
        T.transform = T.transform * (*it);
    }
    T.inverseTransform = glm::inverse(T.transform);
    T.invTranspose = glm::transpose(T.inverseTransform);
    return std::move(T);
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

    const tinygltf::Scene& scene = model->scenes[model->defaultScene];
    for (size_t i = 0; i < scene.nodes.size(); i++)
    {
        const tinygltf::Node& node = model->nodes[scene.nodes[i]];
        loadNode(node);
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
                int materialId = primitive.material;

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



int Scene::loadCamera(const tinygltf::Node& node, const glm::mat4& transform)
{
    cout << "Loading Camera ..." << endl;
    Camera& camera = state.camera;
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
        fovy = perspective.yfov;
        camera.resolution.x = perspective.aspectRatio * camera.resolution.y;
    }
    else if (gltfCamera.type == "orthographic")
    {
        const tinygltf::OrthographicCamera& ortho = gltfCamera.orthographic;
        cout << "Orthographic Camera not implemented." << endl;
        return 0;
    }
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}



int Scene::loadMaterial()
{

    return 1;
}
