// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// GRF's graphics-side object model (Apache-2.0): the resource managers of its GameContext, the
// geometry data, surfaces, textures and vertex buffers they hold, the 3D scene's nodes and objects
// with their graphics interpreters GraphicsGeometry, GraphicsLight and GraphicsCamera, the 2D
// scene's Image2D objects with GraphicsOverlay2D, and GraphicsTask::Render. Lifetimes follow
// GRF's reference counting, so GL objects are created and deleted in GRF's order.

#ifndef CUJU_RENDER_SCENE_SCENE_H
#define CUJU_RENDER_SCENE_SCENE_H

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../assets/assets.h"
#include "renderer3d.h"

namespace cuju::render::scene {
using SurfacePtr = ResourcePtr<assets::Surface>;

struct Material {
    SurfacePtr diffuse, normal, specular, illumination;
    float shininess = 0.0f;
    float specular_amount = 0.0f;
    Vector3 self_illumination;
};

struct Mesh {
    Material material;
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
};

struct GeometryData {
    std::vector<Mesh> meshes;
    bool dynamic = false;
    bool aabb_dirty = false;
    Aabb aabb;
    GeometryData() { aabb.Reset(); }
    GeometryData Copy();
    Aabb GetAabb();
    void AddMesh(Mesh mesh);
};

template <typename T>
class ResourceManager {
public:
    using Loader = std::function<void(const std::string &path, T *resource)>;
    void SetLoader(Loader loader) { loader_ = std::move(loader); }

    ResourcePtr<T> Fetch(const std::string &filename, bool load, bool *already_there,
                         bool use_existing);
    ResourcePtr<T> Fetch(const std::string &filename, bool load = true, bool use_existing = true) {
        bool unused = false;
        return Fetch(filename, load, &unused, use_existing);
    }
    ResourcePtr<T> FetchCopy(const std::string &filename, const std::string &new_name,
                             const std::function<std::unique_ptr<T>(T &)> &copy);
    void RemoveUnused();
    void Clear() { resources_.clear(); }

private:
    void Register(const ResourcePtr<T> &resource);
    std::map<std::string, ResourcePtr<T>> resources_;
    Loader loader_;
};

enum class ObjectType : std::uint8_t { kCamera = 1, kImage2D = 2, kGeometry = 3, kLight = 5 };

class Node;
class Scene;

struct ShadowMap {
    ResourcePtr<Texture> texture;
    int framebuffer = 0;
    std::string camera_name;
    std::deque<VertexBufferQueueEntry> visible;
    Matrix4 projection, view;
};

struct SpatialState {
    Vector3 derived_position;
    Quaternion derived_rotation;
    std::optional<Aabb> aabb;
};

class Object {
public:
    Object(ObjectType type, std::string name) : type(type), name(std::move(name)) {}
    ObjectType type;
    std::string name;
    bool enabled = true;
    Node *parent = nullptr;
    bool absolute = false;
    Vector3 position;
    Quaternion rotation;
    SpatialState spatial;

    ResourcePtr<GeometryData> data;
    bool has_gfx = false;
    ResourcePtr<VertexBuffer> vertex_buffer;
    std::vector<VertexBufferIndex> indices;
    bool uses_indices = false;
    Vector3 gfx_position;
    Quaternion gfx_rotation;

    Vector3 light_color{1, 1, 1};
    float light_radius = 512;
    bool directional = false;
    bool shadow = false;
    std::vector<ShadowMap> shadow_maps;
    std::uint64_t light_key = 0;
    Vector3 light_gfx_position;

    float fov = 0, near_cap = 0, far_cap = 0;
    int view_id = -1;
    Vector3 cam_position;
    Quaternion cam_rotation;
    float cam_fov = 45, cam_near = 0, cam_far = 0;
    struct ViewBuffer {
        std::deque<VertexBufferQueueEntry> geometry;
        std::deque<LightQueueEntry> lights;
        Matrix4 camera_matrix;
        float fov = 0, near_cap = 0, far_cap = 0;
    } view_buffer;
    Vector3 exec_position;
    Quaternion exec_rotation;
    float exec_fov = 0, exec_near = 0, exec_far = 0;
    Vector3 shadow_camera_position;

    Aabb GetAabb();
};
using ObjectPtr = std::shared_ptr<Object>;

class Node {
public:
    explicit Node(std::string name) : name(std::move(name)) {}
    std::string name;
    Node *parent = nullptr;
    bool absolute = false;
    Vector3 position;
    Quaternion rotation;
    std::vector<ObjectPtr> objects;
    std::vector<std::shared_ptr<Node>> nodes;
    std::optional<Aabb> aabb;

    void AddNode(std::shared_ptr<Node> node);
    void AddObject(ObjectPtr object);
    Aabb GetAabb();
    ObjectPtr GetObject(const std::string &name);
    void Collect(std::vector<ObjectPtr> &out, ObjectType type);
    void CullObjects(std::deque<ObjectPtr> &out, const std::vector<Plane> &planes);
    void CullGeometry(std::deque<ObjectPtr> &out, const std::vector<Plane> &planes);
};
using NodePtr = std::shared_ptr<Node>;

struct Image2D {
    std::string name;
    bool enabled = true;
    int poke_priority = 0;
    int position[2] = {0, 0};
    int size[2] = {0, 0};
    SurfacePtr image;
    bool has_gfx = false;
    ResourcePtr<Texture> texture;
    int gfx_position[2] = {0, 0};
    int gfx_size[2] = {0, 0};
};
using Image2DPtr = std::shared_ptr<Image2D>;

struct MediaSource {
    std::string data_dir;
    std::function<bool(const std::string &path, assets::Surface *)> load_surface;
    std::function<bool(const std::string &path, assets::GeometryData *)> load_ase;
};

class Scene {
public:
    Scene(GlStream &gl, int width, int height, MediaSource media);

    Renderer3D &renderer() { return renderer_; }
    GlStream &gl() { return gl_; }
    ResourceManager<GeometryData> &geometry_manager() { return geometry_manager_; }
    ResourceManager<assets::Surface> &surface_manager() { return surface_manager_; }
    ResourceManager<Texture> &texture_manager() { return texture_manager_; }
    ResourceManager<VertexBuffer> &vertices_manager() { return vertices_manager_; }
    Node &root() { return *root_; }
    std::vector<Image2DPtr> &images() { return images2d_; }

    void RemoveUnused();

    // GRF's `game_config.render`.
    void SetRender(bool render) { render_ = render; }
    bool render() const { return render_; }

    ObjectPtr CreateGeometry(const std::string &name);
    void SetGeometryData(Object &object, ResourcePtr<GeometryData> data);
    void OnUpdateGeometryData(Object &object, bool update_materials = true);
    void ExitObject(Object &object);
    void ExitNode(Node &node);
    void DeleteNode(Node &parent, const NodePtr &node);
    void DeleteObject(Node &parent, const ObjectPtr &object);
    void InitCamera(Object &camera);

    Image2DPtr CreateImage2D(const std::string &name, int width, int height, bool scene_register);
    void SetImage(Image2D &image, SurfacePtr surface);
    void ImageChanged(Image2D &image);
    void MoveImage(Image2D &image, int x, int y);
    void DeleteImage(const Image2DPtr &image);

    void Render(bool swap, std::vector<std::uint8_t> *screen);
    std::vector<ObjectPtr> last_visible, last_shadow;

private:
    void LoadMaterials(const Material &material, RendererMaterial *out);
    ResourcePtr<Texture> TextureForSurface(const SurfacePtr &surface, bool diffuse);
    void Execute(Object &camera);
    void EnqueueShadowMap(Object &camera, Object &light);
    void LightEnqueueShadowMap(Object &light, Object &camera, const std::deque<ObjectPtr> &visible);
    void LightPoke(Object &light);
    void CameraEnqueueView(Object &camera, const std::deque<ObjectPtr> &geometry,
                           const std::deque<ObjectPtr> &lights);
    void CameraPoke(Object &camera);
    void GetVertexBufferQueue(Object &object, std::deque<VertexBufferQueueEntry> &queue);

    GlStream &gl_;
    Renderer3D renderer_;
    MediaSource media_;
    ResourceManager<GeometryData> geometry_manager_;
    ResourceManager<assets::Surface> surface_manager_;
    ResourceManager<Texture> texture_manager_;
    ResourceManager<VertexBuffer> vertices_manager_;
    NodePtr root_;
    std::vector<Image2DPtr> images2d_;
    bool render_ = true;
};

}  // namespace cuju::render::scene

#endif
