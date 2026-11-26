// Copyright 2026 The cuju Authors
// Copyright 2019 Google LLC & Bastiaan Konings
// SPDX-License-Identifier: Apache-2.0
//
// ObjectLoader of utils/objectloader.cpp over XMLLoader of utils/xmlloader.cpp (GRF, Apache-2.0).
// An XML tree's children live in a std::multimap keyed by tag, so they are visited sorted by tag,
// equal tags in file order. GetVectorFromString and GetQuaternionFromString are base/utils.cpp's.

#include <map>

#include "assets/assets.h"
#include "assets/text.h"
#include "cuju/quaternion.h"
#include "cuju/vector3.h"

namespace cuju::render::assets {
namespace {
struct XmlTree {
    std::string value;
    std::multimap<std::string, XmlTree> children;
};

bool IsSpace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

bool BuildTree(XmlTree &tree, const std::string &source, std::string *error) {
    std::size_t index = source.find('<', 0);
    if (index == std::string::npos) {
        tree.value.clear();
        for (char c : source)
            if (!IsSpace(c)) tree.value.push_back(c);
        return true;
    }
    while (index != std::string::npos) {
        std::size_t indexEnd = source.find('>', index);
        if (indexEnd == std::string::npos) {
            if (error) *error = "unterminated tag";
            return false;
        }
        const std::string tag = source.substr(index + 1, indexEnd - index - 1);
        index = indexEnd;
        int depth = 1;
        while (depth != 0) {
            const std::size_t open = source.find("<" + tag + ">", indexEnd + 1);
            const std::size_t close = source.find("</" + tag + ">", indexEnd + 1);
            if (open > close || open == std::string::npos) {
                depth--;
                indexEnd = close;
            } else {
                depth++;
                indexEnd = open;
            }
            if (indexEnd == std::string::npos) {
                if (error) *error = "No closing tag found for <" + tag + ">";
                return false;
            }
        }
        XmlTree child;
        if (!BuildTree(child, source.substr(index + 1, indexEnd - index - 1), error)) return false;
        tree.children.insert(std::make_pair(tag, std::move(child)));
        index = source.find('>', indexEnd);
        index = source.find('<', index);
    }
    return true;
}

Vec3 ToVec3(const cuju::Vector3 &v) {
    Vec3 out;
    for (int i = 0; i < 3; i++) out.v[i] = v.coordinates[i];
    return out;
}

cuju::Vector3 VectorFromString(const std::string &s) {
    if (s.empty()) return cuju::Vector3(0.0f);
    std::vector<std::string> tokens;
    text::Tokenize(s, tokens, ",");
    cuju::Vector3 vector;
    if (!tokens.empty()) vector.coordinates[0] = (float)text::Atof(tokens[0]);
    if (tokens.size() > 1) vector.coordinates[1] = (float)text::Atof(tokens[1]);
    if (tokens.size() > 2) vector.coordinates[2] = (float)text::Atof(tokens[2]);
    return vector;
}

// LoadObject's offset is always Vector3(0), and adding it turns -0 into +0.
Vec3 PositionFromString(const std::string &s) {
    return ToVec3(VectorFromString(s) + cuju::Vector3(0.0f));
}

bool QuaternionFromString(const std::string &s, float rotation[4], std::string *error) {
    std::vector<std::string> tokens;
    text::Tokenize(s, tokens, ",");
    if (tokens.size() != 4) {
        if (error) *error = "rotation needs four values: " + s;
        return false;
    }
    constexpr float kPi = 3.1415926535897932384626433832795028841972f;
    const float angle = (float)(text::Atof(tokens[0]) / 360.0 * 2.0 * (double)kPi);
    const cuju::Vector3 axis((float)text::Atof(tokens[1]), (float)text::Atof(tokens[2]),
                             (float)text::Atof(tokens[3]));
    cuju::Quaternion quaternion;
    quaternion.SetAngleAxis(angle, axis);
    for (int i = 0; i < 4; i++) rotation[i] = quaternion.elements[i];
    return true;
}

using Children = std::multimap<std::string, XmlTree>;

void Properties(const Children &tree, std::map<std::string, std::string> &properties) {
    for (const auto &property : tree) properties[property.first] = property.second.value;
}

std::string Property(const std::map<std::string, std::string> &properties, const char *name) {
    const auto found = properties.find(name);
    return found == properties.end() ? std::string() : found->second;
}

bool LoadObjectImpl(const std::string &nodename, const XmlTree &objectTree, ObjectNode *node,
                    std::string *error) {
    node->name = "objectnode: " + nodename;
    const std::string dirpart = nodename.substr(0, nodename.find_last_of('/') + 1);
    for (const auto &object : objectTree.children) {
        const std::string &key = object.first;
        const XmlTree &tree = object.second;
        if (key == "node") {
            auto child = std::make_unique<ObjectNode>();
            if (!LoadObjectImpl(dirpart, tree, child.get(), error)) return false;
            node->entries.push_back({ObjectNode::Entry::kNode, (int)node->nodes.size()});
            node->nodes.push_back(std::move(child));
        } else if (key == "name") {
            node->name = tree.value;
        } else if (key == "position") {
            node->position = PositionFromString(tree.value);
            node->has_position = true;
        } else if (key == "rotation") {
            if (!QuaternionFromString(tree.value, node->rotation, error)) return false;
            node->has_rotation = true;
        } else if (key == "geometry") {
            ObjectGeometry geometry;
            std::string aseFilename;
            std::map<std::string, std::string> properties;
            for (const auto &item : tree.children) {
                if (item.first == "filename") aseFilename = item.second.value;
                if (item.first == "name") geometry.name = item.second.value;
                if (item.first == "position")
                    geometry.position = PositionFromString(item.second.value);
                if (item.first == "rotation" &&
                    !QuaternionFromString(item.second.value, geometry.rotation, error))
                    return false;
                if (item.first == "properties") Properties(item.second.children, properties);
                if (item.first == "localmode") geometry.absolute = item.second.value == "absolute";
            }
            geometry.filename = dirpart + aseFilename;
            geometry.dynamic = Property(properties, "dynamic") == "true";
            node->entries.push_back({ObjectNode::Entry::kGeometry, (int)node->geometries.size()});
            node->geometries.push_back(std::move(geometry));
        } else if (key == "light") {
            ObjectLight light;
            std::map<std::string, std::string> properties;
            for (const auto &item : tree.children) {
                if (item.first == "name") light.name = item.second.value;
                if (item.first == "position")
                    light.position = ToVec3(VectorFromString(item.second.value));
                if (item.first == "properties") Properties(item.second.children, properties);
                if (item.first == "localmode") light.absolute = item.second.value == "absolute";
            }
            light.color = ToVec3(VectorFromString(Property(properties, "color")));
            const auto radius = properties.find("radius");
            light.radius = radius == properties.end() ? 0.0f : (float)text::Atof(radius->second);
            light.directional = Property(properties, "type") == "directional";
            light.shadow = Property(properties, "shadow") == "true";
            node->entries.push_back({ObjectNode::Entry::kLight, (int)node->lights.size()});
            node->lights.push_back(std::move(light));
        }
    }
    return true;
}

}  // namespace

bool LoadObject(const std::string &data_dir, const std::string &path, ObjectNode *node,
                std::string *error) {
    std::string source;
    if (!text::ReadFile(text::DataPath(data_dir, path), &source, error)) return false;
    XmlTree tree;
    std::string reason;
    if (!BuildTree(tree, source, &reason) || tree.children.empty()) {
        if (error)
            *error = path + ": " + (reason.empty() ? std::string("no root element") : reason);
        return false;
    }
    ObjectNode result;
    if (!LoadObjectImpl(path, tree.children.begin()->second, &result, &reason)) {
        if (error) *error = path + ": " + reason;
        return false;
    }
    *node = std::move(result);
    return true;
}

}  // namespace cuju::render::assets
