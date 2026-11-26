// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "scene.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../assets/text.h"

namespace cuju::render::scene {
namespace {
using namespace gl;

float Clamp(float v, float lo, float hi) {
    if (lo > v) return lo;
    if (hi < v) return hi;
    return v;
}

float NormalizedClamp(float value, float min, float max) {
    float b = Clamp(value, min, max);
    return (b - min) / (max - min);
}

Aabb MeshAabb(const Mesh &mesh) {
    Aabb a;
    a.Reset();
    const std::vector<float> &v = mesh.vertices;
    int size = static_cast<int>(v.size());
    auto take = [&](float x, int i) {
        if (x < a.min.coordinates[i]) a.min.coordinates[i] = x;
        if (x > a.max.coordinates[i]) a.max.coordinates[i] = x;
    };
    if (mesh.indices.empty()) {
        unsigned triangles = static_cast<unsigned>(size) / 5u / 3u / 3u;
        for (unsigned t = 0; t < triangles; t++)
            for (unsigned k = 0; k < 3; k++)
                for (unsigned i = 0; i < 3; i++) take(v[t * 9 + k * 3 + i], static_cast<int>(i));
    } else {
        for (std::size_t t = 0; t < mesh.indices.size() / 3; t++)
            for (unsigned k = 0; k < 3; k++)
                for (unsigned i = 0; i < 3; i++)
                    take(v[mesh.indices[t * 3 + k] + i], static_cast<int>(i));
    }
    return a;
}

constexpr int kElements = 5;

}  // namespace

GeometryData GeometryData::Copy() {
    GeometryData c;
    c.meshes = meshes;
    c.aabb = GetAabb();
    c.aabb_dirty = false;
    return c;
}

Aabb GeometryData::GetAabb() {
    if (aabb_dirty) {
        aabb.Reset();
        for (const Mesh &m : meshes) aabb.Add(MeshAabb(m));
        aabb_dirty = false;
    }
    return aabb;
}

void GeometryData::AddMesh(Mesh mesh) {
    meshes.push_back(std::move(mesh));
    aabb.Reset();
    aabb_dirty = true;
}

template <typename T>
ResourcePtr<T> ResourceManager<T>::Fetch(const std::string &filename, bool load,
                                         bool *already_there, bool use_existing) {
    std::string name = assets::text::FileName(filename);
    if (use_existing) {
        auto it = resources_.find(name);
        if (it != resources_.end()) {
            *already_there = true;
            return it->second;
        }
    }
    *already_there = false;
    auto resource = std::make_shared<Resource<T>>(name);
    if (load && loader_) loader_(filename, resource->res.get());
    Register(resource);
    return resource;
}

template <typename T>
ResourcePtr<T> ResourceManager<T>::FetchCopy(const std::string &filename,
                                             const std::string &new_name,
                                             const std::function<std::unique_ptr<T>(T &)> &copy) {
    auto it = resources_.find(new_name);
    if (it != resources_.end()) return Fetch(new_name, false, true);
    ResourcePtr<T> source = Fetch(filename, true, true);
    auto c = std::make_shared<Resource<T>>(new_name);
    c->res = copy(*source->res);
    Register(c);
    return c;
}

template <typename T>
void ResourceManager<T>::Register(const ResourcePtr<T> &resource) {
    auto it = resources_.find(resource->ident);
    if (it != resources_.end()) resources_.erase(it);
    resources_.emplace(resource->ident, resource);
}

template <typename T>
void ResourceManager<T>::RemoveUnused() {
    for (auto it = resources_.begin(); it != resources_.end();) {
        if (it->second.use_count() == 1)
            it = resources_.erase(it);
        else
            ++it;
    }
}

template class ResourceManager<GeometryData>;
template class ResourceManager<assets::Surface>;
template class ResourceManager<Texture>;
template class ResourceManager<VertexBuffer>;

Aabb Object::GetAabb() {
    if (spatial.aabb) return *spatial.aabb;
    if (type == ObjectType::kGeometry)
        return data->res->GetAabb()
            .Rotated(spatial.derived_rotation)
            .Translated(spatial.derived_position);
    if (type == ObjectType::kLight) {
        Aabb a;
        a.min = spatial.derived_position - light_radius;
        a.max = spatial.derived_position + light_radius;
        return a;
    }
    Aabb a;
    a.Reset();
    return a;
}

void Node::AddNode(std::shared_ptr<Node> node) {
    node->parent = this;
    nodes.push_back(std::move(node));
}

void Node::AddObject(ObjectPtr object) {
    object->parent = this;
    objects.push_back(std::move(object));
}

Aabb Node::GetAabb() {
    if (aabb) return *aabb;
    Aabb a;
    a.Reset();
    for (const NodePtr &n : nodes) a.Add(n->GetAabb());
    for (const ObjectPtr &o : objects) a.Add(o->GetAabb());
    return a;
}

ObjectPtr Node::GetObject(const std::string &n) {
    for (const ObjectPtr &o : objects)
        if (o->name == n) return o;
    return nullptr;
}

void Node::Collect(std::vector<ObjectPtr> &out, ObjectType type) {
    for (const ObjectPtr &o : objects)
        if (o->type == type) out.push_back(o);
    for (const NodePtr &n : nodes) n->Collect(out, type);
}

void Node::CullObjects(std::deque<ObjectPtr> &out, const std::vector<Plane> &planes) {
    for (const ObjectPtr &o : objects)
        if (o->GetAabb().Intersects(planes)) out.push_back(o);
    for (const NodePtr &n : nodes)
        if (n->GetAabb().Intersects(planes)) n->CullObjects(out, planes);
}

void Node::CullGeometry(std::deque<ObjectPtr> &out, const std::vector<Plane> &planes) {
    for (const ObjectPtr &o : objects)
        if (o->type == ObjectType::kGeometry && o->GetAabb().Intersects(planes)) out.push_back(o);
    for (const NodePtr &n : nodes)
        if (n->GetAabb().Intersects(planes)) n->CullGeometry(out, planes);
}

Scene::Scene(GlStream &gl, int width, int height, MediaSource media)
    : gl_(gl), renderer_(gl, width, height), media_(std::move(media)) {
    root_ = std::make_shared<Node>("Scene3D root node");
    surface_manager_.SetLoader(
        [this](const std::string &path, assets::Surface *s) { media_.load_surface(path, s); });
    geometry_manager_.SetLoader([this](const std::string &path, GeometryData *g) {
        assets::GeometryData raw;
        media_.load_ase(path, &raw);
        auto fetch = [this](const std::string &name, const std::string &p) -> SurfacePtr {
            if (name.empty()) return nullptr;
            return surface_manager_.Fetch(p, true, true);
        };
        for (assets::TriangleMesh &m : raw.meshes) {
            Mesh mesh;
            const assets::Material &am = m.material;
            mesh.material.diffuse = fetch(am.diffuse, am.diffuse_path);
            mesh.material.normal = fetch(am.normal, am.normal_path);
            mesh.material.specular = fetch(am.specular, am.specular_path);
            mesh.material.illumination = fetch(am.illumination, am.illumination_path);
            mesh.material.shininess = am.shininess;
            mesh.material.specular_amount = am.specular_amount;
            mesh.material.self_illumination = Vector3(
                am.self_illumination.v[0], am.self_illumination.v[1], am.self_illumination.v[2]);
            mesh.vertices = std::move(m.vertices);
            mesh.indices = std::move(m.indices);
            g->AddMesh(std::move(mesh));
        }
    });
}

void Scene::RemoveUnused() {
    geometry_manager_.RemoveUnused();
    surface_manager_.RemoveUnused();
    texture_manager_.RemoveUnused();
    vertices_manager_.RemoveUnused();
}

ObjectPtr Scene::CreateGeometry(const std::string &name) {
    auto o = std::make_shared<Object>(ObjectType::kGeometry, name);
    o->has_gfx = true;
    return o;
}

ResourcePtr<Texture> Scene::TextureForSurface(const SurfacePtr &surface, bool diffuse) {
    bool already = false;
    ResourcePtr<Texture> t = texture_manager_.Fetch(surface->ident, false, &already, true);
    if (!already) {
        const assets::Surface &image = *surface->res;
        t->res->SetRenderer(&renderer_);
        if (diffuse) {
            bool alpha = image.alpha;
            t->res->CreateTexture(alpha ? InternalFormat::kSRGBA8 : InternalFormat::kSRGB8,
                                  alpha ? PixelFormat::kRGBA : PixelFormat::kRGB, image.w, image.h,
                                  alpha, true, true, true);
            t->res->UpdateTexture(image, alpha, true);
        } else {
            t->res->CreateTexture(InternalFormat::kRGB8, PixelFormat::kRGB, image.w, image.h, false,
                                  true, true, true);
            t->res->UpdateTexture(image, false, true);
        }
    }
    return t;
}

void Scene::LoadMaterials(const Material &material, RendererMaterial *out) {
    ResourcePtr<Texture> diffuse, normal, specular, illumination;
    if (material.diffuse) diffuse = TextureForSurface(material.diffuse, true);
    if (material.normal) normal = TextureForSurface(material.normal, false);
    if (material.specular) specular = TextureForSurface(material.specular, false);
    if (material.illumination) illumination = TextureForSurface(material.illumination, false);
    if (diffuse) out->diffuse = diffuse;
    if (normal) out->normal = normal;
    if (specular) out->specular = specular;
    if (illumination) out->illumination = illumination;
    out->shininess = material.shininess;
    out->specular_amount = material.specular_amount;
    out->self_illumination = material.self_illumination;
}

void Scene::SetGeometryData(Object &object, ResourcePtr<GeometryData> data) {
    object.data = std::move(data);
    if (!object.has_gfx) return;
    GeometryData &g = *object.data->res;
    bool dynamic = g.dynamic;
    bool already = false;
    object.vertex_buffer = vertices_manager_.Fetch(object.data->ident, false, &already, true);
    std::vector<Mesh> meshes = g.meshes;
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
    int vertices_data_size = 0;
    if (!already) {
        std::size_t indices_size = 0;
        for (const Mesh &m : meshes) {
            vertices_data_size += static_cast<int>(m.vertices.size());
            indices_size += m.indices.size();
        }
        vertices.resize(vertices_data_size);
        indices.reserve(indices_size);
    }
    int start = 0, current = 0;
    for (const Mesh &m : meshes) {
        RendererMaterial r3d;
        LoadMaterials(m.material, &r3d);
        start += current;
        current = 0;
        int mesh_size = static_cast<int>(m.vertices.size());
        if (!already) {
            for (int e = 0; e < kElements; e++)
                std::memcpy(&vertices[e * (vertices_data_size / kElements) + start],
                            &m.vertices[e * (mesh_size / kElements)],
                            (mesh_size / kElements) * sizeof(float));
            for (std::uint32_t idx : m.indices) indices.push_back(start / 3 + idx);
        }
        current = mesh_size / kElements;
        VertexBufferIndex vbi;
        if (!indices.empty()) {
            vbi.start = static_cast<int>(indices.size() - m.indices.size());
            vbi.size = static_cast<int>(m.indices.size());
        } else {
            vbi.start = start / 3;
            vbi.size = current / 3;
        }
        vbi.material = r3d;
        object.indices.push_back(vbi);
    }
    object.uses_indices = !indices.empty();
    if (!already) {
        object.vertex_buffer->res->SetTriangleMesh(vertices, vertices_data_size, indices);
        object.vertex_buffer->res->CreateOrUpdateVertexBuffer(&renderer_, dynamic);
    }
}

void Scene::OnUpdateGeometryData(Object &object, bool update_materials) {
    // GRF: Geometry::OnUpdateGeometryData tells its observers only while rendering.
    if (!render_ || !object.has_gfx) return;
    GeometryData &g = *object.data->res;
    bool dynamic = g.dynamic;
    const std::vector<Mesh> &meshes = g.meshes;
    VertexBuffer &vb = *object.vertex_buffer->res;
    int current_size = vb.vertices_data_size();
    std::vector<float> array;
    float *vertices = nullptr;
    std::vector<std::uint32_t> indices;
    int vertices_data_size = 0;
    std::size_t indices_size = 0;
    for (const Mesh &m : meshes) {
        vertices_data_size += static_cast<int>(m.vertices.size());
        indices_size += m.indices.size();
    }
    indices.reserve(indices_size);
    bool new_float_data, update_indices;
    if (vertices_data_size == current_size) {
        vertices = vb.vertices().data();
        new_float_data = false;
        update_indices = update_materials;
    } else {
        array.resize(vertices_data_size);
        vertices = array.data();
        new_float_data = true;
        update_indices = true;
    }
    int start = 0, start_indices = 0, current = 0;
    if (update_indices) object.indices.clear();
    for (const Mesh &m : meshes) {
        start += current;
        current = 0;
        int mesh_size = static_cast<int>(m.vertices.size());
        for (int e = 0; e < kElements; e++)
            std::memcpy(&vertices[e * (vertices_data_size / kElements) + start],
                        &m.vertices[e * (mesh_size / kElements)],
                        (mesh_size / kElements) * sizeof(float));
        if (!object.uses_indices)
            for (std::uint32_t idx : m.indices) indices.push_back(start / 3 + idx);
        current = mesh_size / kElements;
        if (update_indices) {
            RendererMaterial r3d;
            LoadMaterials(m.material, &r3d);
            VertexBufferIndex vbi;
            vbi.material = r3d;
            if ((!object.uses_indices && !indices.empty()) || object.uses_indices) {
                vbi.start = start_indices;
                vbi.size = static_cast<int>(m.indices.size());
            } else {
                vbi.start = start / 3;
                vbi.size = current / 3;
            }
            start_indices += static_cast<int>(m.indices.size());
            object.indices.push_back(vbi);
        }
    }
    if (!indices.empty()) object.uses_indices = true;
    if (new_float_data)
        vb.SetTriangleMesh(array, vertices_data_size, indices);
    else
        vb.TriangleMeshWasUpdatedExternally(vertices_data_size, indices);
    vb.CreateOrUpdateVertexBuffer(&renderer_, dynamic);
}

void Scene::ExitObject(Object &object) {
    if (object.type == ObjectType::kGeometry) {
        object.vertex_buffer.reset();
        object.indices.clear();
        object.has_gfx = false;
        object.data.reset();
    } else if (object.type == ObjectType::kCamera) {
        if (object.view_id >= 0) renderer_.DeleteView(object.view_id);
        object.view_id = -1;
    } else if (object.type == ObjectType::kLight) {
        for (ShadowMap &m : object.shadow_maps) {
            renderer_.BindFrameBuffer(m.framebuffer);
            renderer_.SetFrameBufferTexture2D(Attachment::kDepth, 0);
            renderer_.DeleteFrameBuffer(m.framebuffer);
            m.visible.clear();
            m.texture.reset();
        }
        object.shadow_maps.clear();
    }
}

void Scene::ExitNode(Node &node) {
    for (const ObjectPtr &o : node.objects) ExitObject(*o);
    node.objects.clear();
    for (const NodePtr &n : node.nodes) ExitNode(*n);
    node.nodes.clear();
}

void Scene::DeleteNode(Node &parent, const NodePtr &node) {
    auto it = std::find(parent.nodes.begin(), parent.nodes.end(), node);
    if (it != parent.nodes.end()) {
        NodePtr keep = *it;
        ExitNode(*keep);
        parent.nodes.erase(it);
    }
}

void Scene::DeleteObject(Node &parent, const ObjectPtr &object) {
    auto it = std::find(parent.objects.begin(), parent.objects.end(), object);
    if (it != parent.objects.end()) {
        ExitObject(**it);
        (*it)->parent = nullptr;
        parent.objects.erase(it);
    }
}

void Scene::InitCamera(Object &camera) { camera.view_id = renderer_.CreateView(0, 0, 100, 100); }

Image2DPtr Scene::CreateImage2D(const std::string &name, int width, int height,
                                bool scene_register) {
    SurfacePtr surface = surface_manager_.Fetch(name, false, true);
    assets::Surface &s = *surface->res;
    s.w = width;
    s.h = height;
    s.pitch = width * 4;
    s.alpha = true;
    s.pixels.assign(static_cast<std::size_t>(width) * height * 4, 0);
    auto image = std::make_shared<Image2D>();
    image->name = name;
    image->has_gfx = scene_register;
    SetImage(*image, surface);
    MoveImage(*image, renderer_.context_width(), renderer_.context_height());
    if (scene_register) {
        image->enabled = false;
        images2d_.push_back(image);
    }
    return image;
}

void Scene::SetImage(Image2D &image, SurfacePtr surface) {
    image.image = std::move(surface);
    image.position[0] = 0;
    image.position[1] = 0;
    image.size[0] = image.image->res->w;
    image.size[1] = image.image->res->h;
    if (!image.has_gfx) return;
    bool already = false;
    image.texture = texture_manager_.Fetch(image.image->ident, false, &already, true);
    const assets::Surface &s = *image.image->res;
    if (!already) {
        image.texture->res->SetRenderer(&renderer_);
        image.texture->res->CreateTexture(InternalFormat::kRGBA8,
                                          s.alpha ? PixelFormat::kRGBA : PixelFormat::kRGB, s.w,
                                          s.h, s.alpha, false, false, false);
        image.texture->res->UpdateTexture(s, s.alpha, false);
        image.gfx_position[0] = 0;
        image.gfx_position[1] = 0;
        image.gfx_size[0] = s.w;
        image.gfx_size[1] = s.h;
    } else {
        ImageChanged(image);
    }
}

void Scene::ImageChanged(Image2D &image) {
    if (!image.has_gfx) return;
    const assets::Surface &s = *image.image->res;
    if (image.gfx_size[0] != s.w || image.gfx_size[1] != s.h) {
        image.gfx_size[0] = s.w;
        image.gfx_size[1] = s.h;
        image.texture->res->ResizeTexture(s, InternalFormat::kRGBA8,
                                          s.alpha ? PixelFormat::kRGBA : PixelFormat::kRGB, s.alpha,
                                          false);
    } else {
        image.texture->res->UpdateTexture(s, s.alpha, false);
    }
}

void Scene::MoveImage(Image2D &image, int x, int y) {
    image.position[0] = x;
    image.position[1] = y;
    if (image.has_gfx) {
        image.gfx_position[0] = x;
        image.gfx_position[1] = y;
    }
}

void Scene::DeleteImage(const Image2DPtr &image) {
    for (auto it = images2d_.begin(); it != images2d_.end();) {
        if (*it == image) {
            (*it)->texture.reset();
            (*it)->has_gfx = false;
            (*it)->image.reset();
            it = images2d_.erase(it);
        } else {
            ++it;
        }
    }
}

void Scene::GetVertexBufferQueue(Object &object, std::deque<VertexBufferQueueEntry> &queue) {
    VertexBufferQueueEntry e;
    e.indices = &object.indices;
    e.buffer = object.vertex_buffer;
    e.position = object.gfx_position;
    e.rotation = object.gfx_rotation;
    queue.push_back(std::move(e));
}

void Scene::Render(bool swap, std::vector<std::uint8_t> *screen) {
    std::stable_sort(images2d_.begin(), images2d_.end(),
                     [](const Image2DPtr &a, const Image2DPtr &b) {
                         return a->poke_priority < b->poke_priority;
                     });
    std::vector<Overlay2DQueueEntry> queue;
    for (const Image2DPtr &im : images2d_) {
        if (!im->enabled || !im->has_gfx) continue;
        Overlay2DQueueEntry q;
        q.texture = im->texture->res->id();
        q.position[0] = im->gfx_position[0];
        q.position[1] = im->gfx_position[1];
        q.size[0] = im->gfx_size[0];
        q.size[1] = im->gfx_size[1];
        queue.push_back(q);
    }
    last_visible.clear();
    last_shadow.clear();
    std::vector<ObjectPtr> cameras;
    root_->Collect(cameras, ObjectType::kCamera);
    for (const ObjectPtr &c : cameras)
        if (c->enabled) Execute(*c);
    std::function<void(Node &, ObjectType)> poke = [&](Node &node, ObjectType type) {
        for (std::size_t i = 0; i < node.objects.size(); ++i) {
            Object &o = *node.objects[i];
            if (!o.enabled || o.type != type) continue;
            if (type == ObjectType::kLight)
                LightPoke(o);
            else
                CameraPoke(o);
        }
        for (std::size_t i = 0; i < node.nodes.size(); ++i) poke(*node.nodes[i], type);
    };
    poke(*root_, ObjectType::kLight);
    poke(*root_, ObjectType::kCamera);
    renderer_.RenderOverlay2D(queue);
    if (swap) renderer_.SwapBuffers(screen);
}

void Scene::Execute(Object &camera) {
    Vector3 pos = camera.exec_position;
    Quaternion rot = camera.exec_rotation;
    float near_cap = camera.exec_near, far_cap = camera.exec_far;
    float fov = camera.exec_fov * 2.0f;
    float wide = 2.5f;
    std::vector<Plane> planes;
    planes.push_back(
        {pos + rot * Vector3(0, 0, -near_cap), rot * Vector3(0, 0, -1).GetNormalized()});
    planes.push_back({pos, rot * Vector3(0, (3.6f * wide) / (fov / 24.0f), -1).GetNormalized()});
    planes.push_back({pos, rot * Vector3(0, (-3.6f * wide) / (fov / 24.0f), -1).GetNormalized()});
    planes.push_back({pos, rot * Vector3(2.4f / (fov / 24.0f), 0, -1).GetNormalized()});
    planes.push_back({pos, rot * Vector3(-2.4f / (fov / 24.0f), 0, -1).GetNormalized()});
    planes.push_back({pos + rot * Vector3(0, 0, -far_cap), rot * Vector3(0, 0, 1).GetNormalized()});
    std::deque<ObjectPtr> visible;
    root_->CullObjects(visible, planes);
    std::deque<ObjectPtr> geometry, lights;
    for (const ObjectPtr &o : visible) {
        if (!o->enabled) continue;
        if (o->type == ObjectType::kGeometry) geometry.push_back(o);
        if (o->type == ObjectType::kLight) lights.push_back(o);
    }
    last_visible.assign(geometry.begin(), geometry.end());
    for (const ObjectPtr &l : lights)
        if (l->enabled && l->shadow) EnqueueShadowMap(camera, *l);
    CameraEnqueueView(camera, geometry, lights);
}

void Scene::EnqueueShadowMap(Object &camera, Object &light) {
    std::vector<Plane> planes;
    planes.push_back({Vector3(0, -40, 0), Vector3(0, 1, 0.3f).GetNormalized()});
    planes.push_back({Vector3(0, 40, 0), Vector3(0, -1, 0.3f).GetNormalized()});
    planes.push_back({Vector3(-60, 0, 0), Vector3(1, 0, 0.3f).GetNormalized()});
    planes.push_back({Vector3(60, 0, 0), Vector3(-1, 0, 0.3f).GetNormalized()});
    std::deque<ObjectPtr> casters;
    root_->CullGeometry(casters, planes);
    last_shadow.assign(casters.begin(), casters.end());
    LightEnqueueShadowMap(light, camera, casters);
}

void Scene::LightEnqueueShadowMap(Object &light, Object &camera,
                                  const std::deque<ObjectPtr> &visible) {
    if (!light.shadow) return;
    int index = -1;
    for (std::size_t i = 0; i < light.shadow_maps.size(); i++)
        if (light.shadow_maps[i].camera_name == camera.name) index = static_cast<int>(i);
    if (index == -1) {
        ShadowMap map;
        map.texture =
            texture_manager_.Fetch(camera.name + std::to_string(light.light_key), false, false);
        if (map.texture->res->id() == -1) {
            map.texture->res->SetRenderer(&renderer_);
            map.texture->res->CreateTexture(InternalFormat::kDepth16, PixelFormat::kDepth, 2048,
                                            2048, false, false, false, true, true);
            map.framebuffer = renderer_.CreateFrameBuffer();
            renderer_.BindFrameBuffer(map.framebuffer);
            renderer_.SetFrameBufferTexture2D(Attachment::kDepth, map.texture->res->id());
            renderer_.SetRenderTargets({Attachment::kNone});
            renderer_.CheckFrameBufferStatus();
            renderer_.BindFrameBuffer(0);
            renderer_.SetRenderTargets({Attachment::kBack});
        }
        map.camera_name = camera.name;
        light.shadow_maps.push_back(std::move(map));
        index = static_cast<int>(light.shadow_maps.size()) - 1;
    }
    ShadowMap &map = light.shadow_maps[index];
    for (const ObjectPtr &g : visible) GetVertexBufferQueue(*g, map.visible);
    float left = -75, right = 75, bottom = -75, top = 75, near_cap = -150, far_cap = 150;
    Matrix4 proj;
    proj.e[0] = 2 / (right - left);
    proj.e[5] = 2 / (top - bottom);
    proj.e[10] = -2 / (far_cap - near_cap);
    proj.e[3] = -((right + left) / (right - left));
    proj.e[7] = -((top + bottom) / (top - bottom));
    proj.e[11] = -((far_cap + near_cap) / (far_cap - near_cap));
    proj.e[15] = 1;
    map.projection = Matrix4::Identity() * proj;
    Vector3 lp = light.light_gfx_position;
    Vector3 dir = lp.GetNormalized(Vector3(0, 0, -1));
    Vector3 pos = dir + camera.shadow_camera_position.Get2D() + Vector3(0, 70, 0);
    Quaternion rot = QuaternionFromDirection(dir);
    map.view.ConstructInverse(pos, Vector3(1, 1, 1), rot);
}

void Scene::LightPoke(Object &light) {
    if (!light.shadow) return;
    for (ShadowMap &map : light.shadow_maps) {
        renderer_.UseShader("zphase");
        renderer_.BindFrameBuffer(map.framebuffer);
        renderer_.SetViewport(0, 0, map.texture->res->width(), map.texture->res->height());
        renderer_.ClearBuffer(true, false);
        renderer_.SetMatrix("projectionMatrix", map.projection);
        renderer_.SetMatrix("viewMatrix", map.view);
        renderer_.SetRenderTargets({Attachment::kNone});
        renderer_.SetCullingMode(Cull::kFront);
        renderer_.SetBlending(false);
        renderer_.SetDepthFunction(GL_LESS);
        renderer_.SetDepthTesting(true);
        renderer_.SetDepthMask(true);
        renderer_.RenderVertexBuffer(map.visible, RenderMode::kGeometryOnly);
        renderer_.BindFrameBuffer(0);
        renderer_.UseShader("");
        renderer_.SetRenderTargets({Attachment::kBack});
        renderer_.SetViewport(0, 0, renderer_.context_width(), renderer_.context_height());
        renderer_.SetCullingMode(Cull::kBack);
        map.visible.clear();
    }
}

void Scene::CameraEnqueueView(Object &camera, const std::deque<ObjectPtr> &geometry,
                              const std::deque<ObjectPtr> &lights) {
    Object::ViewBuffer &buffer = camera.view_buffer;
    for (const ObjectPtr &g : geometry) GetVertexBufferQueue(*g, buffer.geometry);
    for (const ObjectPtr &lp : lights) {
        Object &l = *lp;
        LightQueueEntry e;
        e.has_shadow = false;
        if (l.shadow) {
            for (ShadowMap &m : l.shadow_maps) {
                if (m.camera_name == camera.name) {
                    e.shadow_texture = m.texture->res->id();
                    e.light_projection = m.projection;
                    e.light_view = m.view;
                    e.has_shadow = true;
                    break;
                }
            }
        }
        e.position = l.spatial.derived_position;
        e.type = l.directional ? 0 : 1;
        e.shadow = l.shadow;
        e.color = l.light_color;
        e.radius = l.light_radius;
        buffer.lights.push_back(e);
    }
    buffer.camera_matrix.ConstructInverse(camera.cam_position, Vector3(1, 1, 1),
                                          camera.cam_rotation);
    buffer.fov = camera.cam_fov;
    buffer.near_cap = camera.cam_near;
    buffer.far_cap = camera.cam_far;
}

void Scene::CameraPoke(Object &camera) {
    Renderer3D &r = renderer_;
    Object::ViewBuffer &buffer = camera.view_buffer;
    r.ClearBuffer(false, true);
    View view = r.GetView(camera.view_id);
    Matrix4 projection = r.CreatePerspectiveMatrix(
        static_cast<float>(view.width) / (static_cast<float>(view.height) * 1.0f), buffer.near_cap,
        buffer.far_cap);
    Matrix4 view_matrix = buffer.camera_matrix;
    int width = r.context_width(), height = r.context_height();
    r.SetViewport(view.x, height - view.y - view.height, view.width, view.height);
    r.SetFOV(buffer.fov);
    float depth_near = 0, depth_far = 0;
    volatile float division = buffer.near_cap - buffer.far_cap;
    if (std::fabs(static_cast<float>(division)) > kEpsilon) {
        division = 1 / division;
        depth_far = (buffer.far_cap * buffer.near_cap) * division;
    }
    if (std::fabs(static_cast<float>(division)) > kEpsilon)
        depth_near = buffer.far_cap / (buffer.far_cap - buffer.near_cap);
    r.UseShader("simple");
    r.BindFrameBuffer(view.gbuffer);
    r.SetViewport(0, 0, view.width, view.height);
    r.SetMatrix("projectionMatrix", projection);
    r.SetMatrix("viewMatrix", view_matrix);
    r.SetRenderTargets({Attachment::kColor0, Attachment::kColor1, Attachment::kColor2});
    r.SetCullingMode(Cull::kBack);
    r.SetBlending(false);
    r.SetDepthTesting(true);
    r.SetDepthFunction(GL_LESS);
    r.SetDepthMask(true);
    r.ClearBuffer(true, true);
    r.RenderVertexBuffer(buffer.geometry, RenderMode::kFull);
    r.BindFrameBuffer(view.accum);
    r.SetRenderTargets({Attachment::kColor0, Attachment::kColor1});
    r.SetTextureUnit(1);
    r.BindTexture(view.gbuffer_normal);
    r.SetTextureUnit(2);
    r.BindTexture(view.gbuffer_depth);
    r.SetTextureUnit(3);
    r.BindTexture(view.gbuffer_aux);
    r.SetTextureUnit(0);
    r.BindTexture(view.gbuffer_albedo);
    r.UseShader("ambient");
    r.SetUniformFloat("ambient", "contextWidth", static_cast<float>(view.width));
    r.SetUniformFloat("ambient", "contextHeight", static_cast<float>(view.height));
    r.SetUniformFloat("ambient", "contextX", 0.0f);
    r.SetUniformFloat("ambient", "contextY", 0.0f);
    r.SetUniformFloat2("ambient", "cameraClip", depth_near, depth_far);
    Matrix4 inverse_pv = (projection * view_matrix).GetInverse();
    r.SetUniformMatrix4("ambient", "inverseProjectionViewMatrix", inverse_pv);
    r.SetUniformMatrix4("ambient", "projectionMatrix", projection);
    r.SetUniformMatrix4("ambient", "viewMatrix", view_matrix);
    r.SetDepthTesting(false);
    r.SetDepthMask(false);
    r.RenderOverlay2D();
    r.UseShader("lighting");
    r.SetBlending(true);
    r.SetBlendingFunction(GL_ONE, GL_ONE);
    r.SetUniformFloat("lighting", "contextWidth", static_cast<float>(view.width));
    r.SetUniformFloat("lighting", "contextHeight", static_cast<float>(view.height));
    r.SetUniformFloat("lighting", "contextX", 0.0f);
    r.SetUniformFloat("lighting", "contextY", 0.0f);
    r.SetUniformMatrix4("lighting", "inverseProjectionViewMatrix", inverse_pv);
    r.SetUniformMatrix4("lighting", "projectionMatrix", projection);
    r.SetUniformMatrix4("lighting", "viewMatrix", view_matrix);
    r.SetDepthTesting(false);
    r.SetDepthMask(false);
    r.RenderLights(buffer.lights, projection, view_matrix);
    r.SetBlending(false);
    r.SetDepthMask(true);
    r.SetCullingMode(Cull::kOff);
    r.SetTextureUnit(1);
    r.BindTexture(0);
    r.SetTextureUnit(2);
    r.BindTexture(0);
    r.SetTextureUnit(3);
    r.BindTexture(0);
    r.SetTextureUnit(0);
    r.BindTexture(0);
    r.BindFrameBuffer(0);
    r.SetRenderTargets({Attachment::kBack});
    r.UseShader("postprocess");
    r.SetUniformFloat("postprocess", "contextWidth", static_cast<float>(view.width));
    r.SetUniformFloat("postprocess", "contextHeight", static_cast<float>(view.height));
    r.SetUniformFloat("postprocess", "contextX", static_cast<float>(view.x));
    r.SetUniformFloat("postprocess", "contextY",
                      static_cast<float>(height - (view.y + view.height)));
    r.SetUniformFloat2("postprocess", "cameraClip", depth_near, depth_far);
    r.SetUniformFloat("postprocess", "fogScale",
                      0.8f - NormalizedClamp(buffer.fov, 20, 100) * 0.6f);
    r.SetViewport(view.x, height - (view.y + view.height), view.width, view.height);
    r.SetTextureUnit(2);
    r.BindTexture(view.gbuffer_depth);
    r.SetTextureUnit(1);
    r.BindTexture(view.accum_modifier);
    r.SetTextureUnit(0);
    r.BindTexture(view.accum_accum);
    r.SetDepthTesting(false);
    r.SetDepthMask(false);
    r.SetFramebufferGammaCorrection(true);
    r.RenderOverlay2D();
    r.SetFramebufferGammaCorrection(false);
    r.SetTextureUnit(2);
    r.BindTexture(0);
    r.SetTextureUnit(1);
    r.BindTexture(0);
    r.SetTextureUnit(0);
    r.BindTexture(0);
    r.UseShader("");
    r.SetRenderTargets({Attachment::kBack});
    r.SetViewport(0, 0, width, height);
    buffer.geometry.clear();
    buffer.lights.clear();
}

}  // namespace cuju::render::scene
