// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0

#include "match.h"

#include <cmath>

#include "../assets/text.h"
#include "cuju/libm/tan_double.h"
#include "random.h"

namespace cuju::render::scene {
namespace {
Vector3 V(const assets::Vec3 &v) { return Vector3(v.v[0], v.v[1], v.v[2]); }

NodePtr BuildNode(Context &context, const assets::ObjectNode &src) {
    Scene &scene = context.scene();
    auto node = std::make_shared<Node>(src.name);
    if (src.has_position) node->position = V(src.position);
    if (src.has_rotation)
        node->rotation =
            Quaternion(src.rotation[0], src.rotation[1], src.rotation[2], src.rotation[3]);
    for (const assets::ObjectNode::Entry &e : src.entries) {
        if (e.kind == assets::ObjectNode::Entry::kNode) {
            node->AddNode(BuildNode(context, *src.nodes[e.index]));
        } else if (e.kind == assets::ObjectNode::Entry::kGeometry) {
            const assets::ObjectGeometry &g = src.geometries[e.index];
            ResourcePtr<GeometryData> data = scene.geometry_manager().Fetch(g.filename, true, true);
            ObjectPtr object = scene.CreateGeometry(g.name);
            if (g.dynamic) data->res->dynamic = true;
            object->absolute = g.absolute;
            object->position = V(g.position);
            object->rotation =
                Quaternion(g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]);
            scene.SetGeometryData(*object, data);
            node->AddObject(object);
        } else {
            const assets::ObjectLight &l = src.lights[e.index];
            auto object = std::make_shared<Object>(ObjectType::kLight, l.name);
            object->absolute = l.absolute;
            object->light_color = V(l.color);
            object->light_radius = l.radius;
            object->directional = l.directional;
            object->shadow = l.shadow;
            object->position = V(l.position);
            node->AddObject(object);
        }
    }
    return node;
}

NodePtr SplitGeometry(Scene &scene, Object &source, float grid) {
    auto result = std::make_shared<Node>(source.name);
    struct Cell {
        float x, y;
        ResourcePtr<GeometryData> data;
    };
    std::vector<Cell> cells;
    std::vector<Mesh> meshes = source.data->res->meshes;
    for (const Mesh &m : meshes) {
        Mesh unindexed;
        unindexed.vertices = m.vertices;
        Aabb a = [&] {
            Aabb r;
            r.Reset();
            unsigned triangles = static_cast<unsigned>(m.vertices.size()) / 5u / 3u / 3u;
            for (unsigned t = 0; t < triangles; t++)
                for (unsigned k = 0; k < 3; k++)
                    for (unsigned i = 0; i < 3; i++) {
                        float x = m.vertices[t * 9 + k * 3 + i];
                        if (x < r.min.coordinates[i]) r.min.coordinates[i] = x;
                        if (x > r.max.coordinates[i]) r.max.coordinates[i] = x;
                    }
            return r;
        }();
        float cx = static_cast<float>(
            static_cast<double>(a.min.coordinates[0] + a.max.coordinates[0]) / 2.0);
        float cy = static_cast<float>(
            static_cast<double>(a.min.coordinates[1] + a.max.coordinates[1]) / 2.0);
        float x = grid * static_cast<float>(static_cast<int>(std::round(cx / grid)));
        float y = grid * static_cast<float>(static_cast<int>(std::round(cy / grid)));
        std::size_t c = 0;
        while (c < cells.size() && !(cells[c].x == x && cells[c].y == y)) c++;
        if (c == cells.size()) {
            std::string at =
                std::to_string(static_cast<int>(x)) + ", " + std::to_string(static_cast<int>(y));
            cells.push_back({x, y,
                             scene.geometry_manager().Fetch(source.name + " gridGeomData @ " + at,
                                                            false, false)});
        }
        Mesh copy;
        copy.material = m.material;
        copy.vertices = m.vertices;
        cells[c].data->res->AddMesh(std::move(copy));
    }
    for (Cell &cell : cells) {
        std::string at = std::to_string(static_cast<int>(cell.x)) + ", " +
                         std::to_string(static_cast<int>(cell.y));
        ObjectPtr g = scene.CreateGeometry(source.name + " gridGeom @ " + at);
        scene.SetGeometryData(*g, cell.data);
        result->AddObject(g);
    }
    return result;
}

void CollectGeometry(Node &node, std::vector<ObjectPtr> &out) {
    node.Collect(out, ObjectType::kGeometry);
}

}  // namespace

Context::Context(GlStream &gl, int width, int height, const Shared &shared) : shared_(shared) {
    MediaSource media;
    media.data_dir = shared.data_dir;
    std::string dir = shared.data_dir;
    media.load_surface = [dir](const std::string &path, assets::Surface *s) {
        std::string error;
        return assets::LoadSurface(assets::text::DataPath(dir, path), s, &error);
    };
    media.load_ase = [dir](const std::string &path, assets::GeometryData *g) {
        std::string error;
        return assets::LoadAse(dir, path, g, &error);
    };
    if (shared.load_surface) media.load_surface = shared.load_surface;
    if (shared.load_ase) media.load_ase = shared.load_ase;
    scene_ = std::make_unique<Scene>(gl, width, height, media);
}

Context::~Context() = default;

void Context::CreateContext() {
    assets::Surface noise;
    std::string error;
    assets::LoadSurface(assets::text::DataPath(shared_.data_dir, "media/shaders/noise.png"), &noise,
                        &error);
    scene_->renderer().CreateContext(noise);
    std::string dir = shared_.data_dir;
    gui_ = std::make_unique<Gui>(
        *scene_, *shared_.atlas,
        [dir](const std::string &path, overlay2d::SourceImage *out) {
            std::string file = assets::text::DataPath(dir, path);
            file = file.substr(0, file.size() - 4) + ".bmp";
            std::string bytes, error;
            if (!assets::text::ReadFile(file, &bytes, &error)) return false;
            return overlay2d::DecodeBmp(reinterpret_cast<const std::uint8_t *>(bytes.data()),
                                        bytes.size(), *out);
        },
        true);
}

NodePtr Context::LoadObject(const std::string &path) {
    assets::ObjectNode src;
    std::string error;
    assets::LoadObject(shared_.data_dir, path, &src, &error);
    return BuildNode(*this, src);
}

void Context::ResetBegin() {
    scene_->RemoveUnused();
    gui_->SetRecursiveZPriority(gui_->root(), 0);
    gui_->ClearPagePath();
}

void Context::StopMatch() {
    if (match_) {
        match_->Exit();
        match_.reset();
    }
}

void Context::CreateLoadingPage(const std::array<std::string, 2> &names,
                                const std::array<std::string, 2> &logos) {
    gui_->CreateLoadingPage(names, logos);
}

void Context::StartMatch(const MatchLook &look, const SkinFunction &skin) {
    match_ = std::make_unique<Match>(*this, look, skin);
}

Match::Body Match::CreateBody(int team, const PlayerLook &player, const SurfacePtr &kit,
                              Node &target, const SkinFunction &skin) {
    Scene &scene = context_.scene();
    Body body;
    body.team = team;
    body.z_multiplier =
        player.z_multiplier > 0.0f ? player.z_multiplier : (1.0f / 1.92f) * player.height;
    SurfacePtr skin_surface = scene.surface_manager().Fetch(
        "media/objects/players/textures/skin0" + std::to_string(player.skin) + ".png", true, true);
    std::string postfix = std::to_string(context_.player_count++);
    Node &tmpl = *context_.fullbody_template;
    body.fullbody_node = std::make_shared<Node>(tmpl.name + postfix);
    body.fullbody_node->absolute = true;
    for (const ObjectPtr &src : tmpl.objects) {
        ObjectPtr copy = scene.CreateGeometry(src->name);
        copy->absolute = src->absolute;
        copy->position = src->position;
        copy->rotation = src->rotation;
        ResourcePtr<GeometryData> data = scene.geometry_manager().FetchCopy(
            src->data->ident, src->data->ident + postfix,
            [](GeometryData &g) { return std::make_unique<GeometryData>(g.Copy()); });
        scene.SetGeometryData(*copy, data);
        body.fullbody_node->AddObject(copy);
    }
    target.AddNode(body.fullbody_node);
    body.fullbody = body.fullbody_node->GetObject("fullbody");
    GeometryData &data = *body.fullbody->data->res;
    for (Mesh &m : data.meshes) {
        if (m.material.diffuse && m.material.diffuse->ident == "skin.jpg") {
            m.material.diffuse = skin_surface;
            m.material.specular_amount = 0.002f;
            m.material.shininess = 0.2f;
        }
    }
    scene.OnUpdateGeometryData(*body.fullbody);
    std::string kit_ident = "kit_template.png";
    if (kit) {
        for (Mesh &m : data.meshes) {
            if (m.material.diffuse && m.material.diffuse->ident == kit_ident) {
                m.material.diffuse = kit;
                m.material.specular_amount = 0.01f;
                m.material.shininess = 0.01f;
            }
        }
    }
    scene.OnUpdateGeometryData(*body.fullbody);
    {
        assets::GeometryData raw;
        for (Mesh &m : data.meshes) {
            assets::TriangleMesh t;
            t.vertices = m.vertices;
            raw.meshes.push_back(std::move(t));
        }
        assets::GeometryData indexed = assets::IndexFullbody(raw, body.z_multiplier);
        for (std::size_t i = 0; i < data.meshes.size(); i++) {
            data.meshes[i].vertices = std::move(indexed.meshes[i].vertices);
            data.meshes[i].indices = std::move(indexed.meshes[i].indices);
        }
    }
    scene.OnUpdateGeometryData(*body.fullbody);
    // GRF: UpdateFullbodyModel(true), the bake, runs only while rendering.
    if (scene.render()) skin(static_cast<int>(bodies_.size()), *body.fullbody, data);
    scene.OnUpdateGeometryData(*body.fullbody, false);
    ResourcePtr<GeometryData> hair_data = scene.geometry_manager().Fetch(
        "media/objects/players/hairstyles/" + player.hair_style + ".ase", true, true);
    body.hair = scene.CreateGeometry("hairstyle");
    body.hair->absolute = true;
    scene.SetGeometryData(*body.hair, hair_data);
    target.AddObject(body.hair);
    SurfacePtr hair_texture = scene.surface_manager().Fetch(
        "media/objects/players/textures/hair/" + player.hair_color + ".png", true, true);
    for (Mesh &m : hair_data->res->meshes) {
        if (m.material.diffuse) {
            m.material.diffuse = hair_texture;
            m.material.specular_amount = 0.01f;
            m.material.shininess = 0.05f;
        }
    }
    scene.OnUpdateGeometryData(*body.hair);
    return body;
}

Match::Match(Context &context, const MatchLook &look, const SkinFunction &skin)
    : context_(context), look_(look) {
    Scene &scene = context.scene();
    Gui &gui = context.gui();
    dynamic_node_ = std::make_shared<Node>("dynamicNode");
    scene.root().AddNode(dynamic_node_);
    ball_node_ = context.LoadObject("media/objects/balls/generic.object");
    dynamic_node_->AddNode(ball_node_);
    {
        std::vector<ObjectPtr> g;
        CollectGeometry(*ball_node_, g);
        ball_ = g.front();
    }
    if (!context.anim_skeleton)
        context.anim_skeleton = context.LoadObject("media/objects/players/player.object");
    if (!context.fullbody_template)
        context.fullbody_template = context.LoadObject("media/objects/players/fullbody.object");
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? look.first_team : 1 - look.first_team;
        teams_[t].node = std::make_shared<Node>("team node #" + std::to_string(t));
        teams_[t].node->absolute = true;
        dynamic_node_->AddNode(teams_[t].node);
    }
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? look.first_team : 1 - look.first_team;
        const TeamLook &team = look.teams[t];
        for (std::size_t i = 0; i < team.players.size(); i++) {
            const PlayerLook &p = team.players[i];
            int stable_id =
                static_cast<int>(i) +
                (k == 0 ? 0 : static_cast<int>(look.teams[look.first_team].players.size()));
            auto *tag = new CaptionView(&gui, "game_player_name_" + std::to_string(stable_id), 0, 0,
                                        1, 2.0f, p.last_name);
            tag->SetTransparency(0.3f);
            gui.AddView(gui.root(), tag);
            teams_[t].tags.push_back(tag);
            if (static_cast<int>(i) < team.active) {
                std::string kit_file = !p.kit.empty() ? p.kit
                                       : p.goalkeeper
                                           ? "media/objects/players/textures/goalie_kit.png"
                                           : team.kit_url + "_kit_02.png";
                SurfacePtr kit = scene.surface_manager().Fetch(kit_file, true, true);
                Body body = CreateBody(t, p, kit, *teams_[t].node, skin);
                body.tag = tag;
                teams_[t].bodies.push_back(static_cast<int>(bodies_.size()));
                bodies_.push_back(body);
                gui.Show(tag);
            }
        }
    }
    SurfacePtr referee_kit =
        scene.surface_manager().Fetch("media/objects/players/textures/referee_kit.png", true, true);
    for (int i = 0; i < 3; i++) {
        Body body = CreateBody(-1, look.officials[i], referee_kit, *dynamic_node_, skin);
        bodies_.push_back(body);
    }
    {
        ResourcePtr<GeometryData> y =
            scene.geometry_manager().Fetch("media/objects/officials/yellowcard.ase", true, true);
        yellow_ = scene.CreateGeometry("yellowcard");
        scene.SetGeometryData(*yellow_, y);
        yellow_->absolute = true;
        yellow_->position = Vector3(0, 0, -10);
        ResourcePtr<GeometryData> r =
            scene.geometry_manager().Fetch("media/objects/officials/redcard.ase", true, true);
        red_ = scene.CreateGeometry("redCard");
        scene.SetGeometryData(*red_, r);
        red_->absolute = true;
        red_->position = Vector3(0, 0, -10);
        dynamic_node_->AddObject(yellow_);
        dynamic_node_->AddObject(red_);
    }
    camera_ = std::make_shared<Object>(ObjectType::kCamera, "camera");
    scene.InitCamera(*camera_);
    camera_->fov = 25;
    camera_node_ = std::make_shared<Node>("cameraNode");
    camera_node_->AddObject(camera_);
    camera_node_->position = Vector3(40, 0, 100);
    dynamic_node_->AddNode(camera_node_);
    if (!scene.render()) {
        if (!context.stadium_no_render)
            context.stadium_no_render =
                context.LoadObject("media/objects/stadiums/test/pitchonly.object");
        std::vector<ObjectPtr> sources;
        CollectGeometry(*context.stadium_no_render, sources);
        stadium_node_ = std::make_shared<Node>("stadium");
        for (const ObjectPtr &s : sources) {
            NodePtr cells = SplitGeometry(scene, *s, 24);
            cells->absolute = true;
            stadium_node_->AddNode(cells);
        }
        stadium_node_->absolute = true;
        scene.root().AddNode(stadium_node_);
    } else if (!context.stadium) {
        context.stadium = context.LoadObject("media/objects/stadiums/test/test.object");
    }
    if (scene.render()) {
        std::vector<std::string> files = {"ad_3xblast01.bmp",
                                          "ad_altfunc01.bmp",
                                          "ad_blauwprint01.bmp",
                                          "ad_bleep01.bmp",
                                          "ad_brokenbread01.bmp",
                                          "ad_groningen01.bmp",
                                          "ad_indietopia01.bmp",
                                          "ad_omeganorth01.bmp",
                                          "ad_polygon01.bmp",
                                          "ad_polygon02.bmp",
                                          "ad_polygon_flyingdutchman02.bmp",
                                          "ad_polygon_jigsawHD02.bmp",
                                          "ad_stark01.bmp",
                                          "ad_stark02.bmp",
                                          "ad_tenshu01.bmp",
                                          "ad_your_ad_here.bmp"};
        std::vector<SurfacePtr> boards;
        for (const std::string &f : files)
            boards.push_back(
                scene.surface_manager().Fetch("media/textures/adboards/" + f, true, true));
        GrfRng rng(look.seed);
        std::vector<ObjectPtr> geoms;
        CollectGeometry(*context.stadium, geoms);
        for (const ObjectPtr &g : geoms) {
            for (Mesh &m : g->data->res->meshes) {
                if (m.material.diffuse && m.material.diffuse->ident.find("ad_placeholder") == 0) {
                    float r = rng.Range(0, static_cast<float>(boards.size()) - 1.001f);
                    m.material.diffuse =
                        boards.at(static_cast<std::size_t>(static_cast<int>(std::floor(r))));
                    m.material.specular_amount = 0.2f;
                    m.material.shininess = 0.1f;
                }
            }
            scene.OnUpdateGeometryData(*g);
        }
        std::vector<ObjectPtr> sources;
        CollectGeometry(*context.stadium, sources);
        stadium_node_ = std::make_shared<Node>("stadium");
        for (const ObjectPtr &s : sources) {
            NodePtr cells = SplitGeometry(scene, *s, 24);
            cells->absolute = true;
            stadium_node_->AddNode(cells);
        }
        stadium_node_->absolute = true;
        scene.root().AddNode(stadium_node_);
    }
    if (!context.goals) {
        context.goals = context.LoadObject("media/objects/stadiums/goals.object");
        context.goals->absolute = true;
    }
    scene.root().AddNode(context.goals);
    if (scene.render() && !context.pitch_loaded) {
        context.pitch_loaded = true;
        const pitch::PitchTextures &tex = context.shared().pitch();
        const pitch::PitchSizes &z = tex.sizes;
        for (int i = 0; i < 4; i++) {
            std::string n = std::to_string(i + 1);
            bool already = false;
            ResourcePtr<Texture> diffuse =
                scene.texture_manager().Fetch("pitch_0" + n + ".png", false, &already, true);
            ResourcePtr<Texture> specular = scene.texture_manager().Fetch(
                "pitch_specular_0" + n + ".png", false, &already, true);
            ResourcePtr<Texture> normal =
                scene.texture_manager().Fetch("pitch_normal_0" + n + ".png", false, &already, true);
            auto upload = [&](ResourcePtr<Texture> &t, InternalFormat f, int w, int h,
                              const std::vector<std::uint8_t> &px) {
                assets::Surface s;
                s.w = w;
                s.h = h;
                s.pitch = 4 * w;
                s.alpha = true;
                s.pixels = px;
                t->res->DeleteTexture();
                t->res->CreateTexture(f, PixelFormat::kRGB, w, h, false, true, true, true);
                t->res->UpdateTexture(s, false, true);
            };
            upload(diffuse, InternalFormat::kSRGB8, z.diffuseWidth, z.diffuseHeight,
                   tex.diffuse[i]);
            upload(specular, InternalFormat::kRGB8, z.specularWidth, z.specularHeight,
                   tex.specular[i]);
            upload(normal, InternalFormat::kRGB8, z.normalWidth, z.normalHeight, tex.normal[i]);
        }
    }
    sun_node_ = context.LoadObject("media/objects/lighting/generic.object");
    dynamic_node_->AddNode(sun_node_);
    sun_ = sun_node_->GetObject("sun");
    sun_->light_key = context.light_count++;
    sun_->position = look.sun_position;
    sun_->light_radius = 1000000.0f;
    sun_->light_color = look.sun_color;
    radar_ = new Radar(&gui, 38, 78, 24, 18);
    gui.AddView(gui.root(), radar_);
    gui.Show(radar_);
    scoreboard_ = new Scoreboard(&gui, {look.teams[0].short_name, look.teams[1].short_name},
                                 {look.teams[0].logo_url, look.teams[1].logo_url});
    gui.AddView(gui.root(), scoreboard_);
    gui.Show(scoreboard_);
    message_ = new CaptionView(&gui, "game_messages", 0, 0, 80, 8, "");
    message_->SetTransparency(0.3f);
    gui.AddView(gui.root(), message_);
    gui.CreateGamePage();
}

Match::~Match() {
    if (!exited_) Exit();
}

void Match::Exit() {
    exited_ = true;
    Scene &scene = context_.scene();
    Gui &gui = context_.gui();
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? look_.first_team : 1 - look_.first_team;
        for (CaptionView *tag : teams_[t].tags) gui.Exit(tag);
        teams_[t].tags.clear();
        for (int b : teams_[t].bodies) {
            Body &body = bodies_[b];
            scene.DeleteNode(*teams_[t].node, body.fullbody_node);
            scene.DeleteObject(*teams_[t].node, body.hair);
        }
        scene.DeleteNode(*dynamic_node_, teams_[t].node);
    }
    for (Body &body : bodies_) {
        if (body.team >= 0) continue;
        scene.DeleteNode(*dynamic_node_, body.fullbody_node);
        scene.DeleteObject(*dynamic_node_, body.hair);
    }
    scene.DeleteNode(*dynamic_node_, ball_node_);
    gui.Exit(message_);
    message_ = nullptr;
    scene.DeleteNode(scene.root(), dynamic_node_);
    scene.DeleteNode(scene.root(), stadium_node_);
    gui.Exit(radar_);
    radar_ = nullptr;
    gui.Exit(scoreboard_);
    scoreboard_ = nullptr;
}

void Match::UploadBodies(const SkinFunction &skin, const std::function<bool(int body)> &active) {
    std::vector<int> order;
    for (int k = 0; k < 2; k++) {
        int t = k == 0 ? look_.first_team : 1 - look_.first_team;
        for (int b : teams_[t].bodies)
            if (!active || active(b)) order.push_back(b);
    }
    for (std::size_t b = 0; b < bodies_.size(); b++)
        if (bodies_[b].team < 0) order.push_back(static_cast<int>(b));
    for (int b : order) {
        skin(b, *bodies_[b].fullbody, *bodies_[b].fullbody->data->res);
        context_.scene().OnUpdateGeometryData(*bodies_[b].fullbody);
    }
}

Vector3 ProjectedCoord(const Vector3 &pos, const ProjectionCamera &camera, int context_w,
                       int context_h, float aspect2d) {
    Matrix4 rot;
    rot.ConstructInverse(camera.position, Vector3(1, 1, 1), camera.rotation);
    float fov = camera.fov;
    float f = static_cast<float>(1.0 / libm::tan((fov / 360.0 * kPi * 2) / 2.0));
    float aspect = static_cast<float>(context_w) / static_cast<float>(context_h);
    if (aspect2d < aspect) aspect = aspect2d;
    float z_near = 40.0f, z_far = 270.0f;
    Matrix4 persp;
    persp.e[0] = f / aspect;
    persp.e[5] = f;
    persp.e[10] = (z_far + z_near) / (z_near - z_far);
    persp.e[11] = (2 * z_far * z_near) / (z_near - z_far);
    persp.e[14] = -1;
    Matrix4 res = persp * rot;
    float x, y, z, w;
    res.MultiplyVec4(pos.coordinates[0], pos.coordinates[1], pos.coordinates[2], 1, x, y, z, w);
    Vector3 r;
    r.coordinates[0] = x / w;
    r.coordinates[1] = y / w;
    r.coordinates[1] = -r.coordinates[1];
    r.coordinates[1] *= aspect2d / aspect;
    r += 1.0f;
    r *= 0.5f;
    r *= 100.0f;
    return r;
}

void Match::PutTags(const std::vector<TagInput> &tags, const ProjectionCamera &camera) {
    Gui &gui = context_.gui();
    std::size_t k = 0;
    for (int order = 0; order < 2; order++) {
        int t = order == 0 ? look_.first_team : 1 - look_.first_team;
        bool mirror = t == 1;
        for (int b : teams_[t].bodies) {
            const TagInput &in = tags.at(k++);
            CaptionView *tag = bodies_[b].tag;
            if (in.keep) continue;
            if (in.shown) {
                Vector3 p = in.geom_position;
                if (mirror) p *= Vector3(-1, -1, 0);
                p = ProjectedCoord(
                    p + Vector3(0, 0.5f, 2.4f), camera, context_.scene().renderer().context_width(),
                    context_.scene().renderer().context_height(), gui.geometry().aspect());
                float w, h;
                tag->GetSize(&w, &h);
                tag->SetColor(in.color);
                tag->SetOutlineColor(overlay2d::OutlineOf(in.color));
                gui.SetPosition(tag, p.coordinates[0] - w * 0.5f, p.coordinates[1] - h);
                const TeamLook &team = look_.teams[t];
                int index = 0;
                for (int bb : teams_[t].bodies) {
                    if (bb == b) break;
                    index++;
                }
                tag->SetCaption(team.players[index].last_name);
                gui.Show(tag);
            } else {
                gui.Hide(tag);
            }
        }
    }
}

void Match::HideMessage() { context_.gui().Hide(message_); }

void Match::SetGoalCount(int team, int goals) { scoreboard_->SetGoalCount(team, goals); }

void Match::SpamMessage(const std::string &message) {
    Gui &gui = context_.gui();
    message_->SetCaption(message);
    float w = message_->TextWidthPercent();
    gui.SetPosition(message_, 50 - w * 0.5f, 5);
    gui.Show(message_);
}

void Match::ShowMessage() {
    Gui &gui = context_.gui();
    float w = message_->TextWidthPercent();
    gui.SetPosition(message_, 50 - w * 0.5f, 5);
    gui.Show(message_);
}

void Match::PutClock(const std::string &time) { scoreboard_->SetTimeStr(time); }

void Match::UploadNetting(const std::function<void(GeometryData &goals)> &netting) {
    ObjectPtr goals = context_.goals->GetObject("goals");
    netting(*goals->data->res);
    context_.scene().OnUpdateGeometryData(*goals, false);
}

}  // namespace cuju::render::scene
