/*
 * Copyright 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <renderengine/PrismScene.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace android::renderengine::prism {

// SurfaceFlinger appends a "#<sequence>" suffix to every surface name, and the sequence counter
// is global and monotonic: a window whose surface is recreated (app switch, relayout) reappears
// with a different name. The prefix in front of a trailing numeric suffix is the stable part and
// is what placement memory is keyed on in the absence of a WindowManager-provided id.
inline std::string stableWindowId(const std::string& layerName) {
    const size_t hash = layerName.rfind('#');
    if (hash == std::string::npos || hash + 1 == layerName.size()) {
        return layerName;
    }
    const auto suffixStart = static_cast<std::string::difference_type>(hash) + 1;
    const bool numericSuffix = std::all_of(layerName.begin() + suffixStart, layerName.end(),
                                           [](char c) { return c >= '0' && c <= '9'; });
    return numericSuffix ? layerName.substr(0, hash) : layerName;
}

// Key for the renderer's placement state. WindowManager publishes a stable per-activity id in
// layer metadata (key 10001); it distinguishes two windows that share a title and survives
// surface recreation. When that id is absent, the stable prefix of the layer name is used.
inline std::string spatialStateKey(int32_t windowId, const std::string& layerName) {
    if (windowId != 0) {
        return "a" + std::to_string(windowId);
    }
    return stableWindowId(layerName);
}

// Automatic ring layout for the slot'th discovered window. Slot 0 is centered; the remaining
// slots alternate left and right, one spacing further out per pair, and wrap into a lower row
// after four side slots. The quad keeps the buffer's aspect ratio at the scene's panel height.
// PrismOS: the portrait panel has a narrow horizontal FOV (~50°), so side positions are
// clamped to the visible frustum — an unclamped ring pushed the third window entirely off
// screen. When horizontalFovDegrees is 0 the clamp is skipped (legacy behavior).
inline PrismSpatialWindowSettings automaticLayout(uint32_t slot, float bufferWidth,
                                                  float bufferHeight,
                                                  const PrismSceneSettings& scene) {
    const float height = std::max(0.05f, scene.spatialWindowHeightMeters);
    const uint32_t ring = (slot + 1) / 2;
    const float direction = (slot & 1u) == 0u ? 1.0f : -1.0f;
    const uint32_t row = slot == 0 ? 0u : (slot - 1) / 4;
    PrismSpatialWindowSettings settings;
    settings.distanceMeters = std::max(0.25f, scene.spatialWindowDistanceMeters);
    settings.heightMeters = height;
    settings.suppressFlatCopy = scene.spatialWindowSuppressFlatCopy;
    // PrismOS ring layout around the camera: slot 0 dead ahead, the rest spread around
    // the user in yaw steps (Horizon-style). The yaw positions are computed on the unit
    // ring and scaled by the distance, so turning the head finds every window.
    const float yawRad = slot == 0 ? 0.0f
            : direction * static_cast<float>(ring) * 30.0f * static_cast<float>(M_PI) / 180.0f;
    settings.yawDegrees = slot == 0 ? 0.0f : direction * static_cast<float>(ring) * 30.0f;
    settings.centerXMeters = slot == 0 ? 0.0f : std::sin(yawRad) * settings.distanceMeters;
    settings.centerYMeters =
            -static_cast<float>(row) * (height + scene.spatialWindowSpacingMeters);
    settings.distanceMeters = slot == 0 ? settings.distanceMeters
            : settings.distanceMeters * std::cos(yawRad);
    return settings;
}

} // namespace android::renderengine::prism
