/*
 * Copyright (C) 2025 The Android Open Source Project
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

#include <math/mat4.h>

#include <cstdint>
#include <string>
#include <vector>

namespace android::renderengine {

// Placement and presentation of one Android surface in the Prism world. `id` is the layer name
// provided by SurfaceFlinger. An entry in PrismSceneSettings::spatialWindows overrides the automatic
// placement for that one id; it never owns a SurfaceControl, GraphicBuffer, or texture.
struct PrismSpatialWindowSettings {
    std::string id;
    float centerXMeters = 0.0f;
    float centerYMeters = 0.0f;
    float distanceMeters = 2.0f;
    float heightMeters = 0.9f;
    bool suppressFlatCopy = true;
};

// Immutable, frame-local parameters for the optional GLES Prism scene. The render target and eye
// viewports are derived from DisplaySettings. Spatial-window layer pointers and EGL resources are
// deliberately not part of this structure: RenderEngine resolves current frame layers itself.
// PrismOS test environment: a world-locked equirect sky, an optional RGBA menu strip, and a
// set of textured world-space quads ("posters") loaded as one packed scene file.
struct PrismEnvQuad {
    std::string name;          // unique within the scene, used by vrender to address the quad
    std::string texturePath;   // raw texture file (PRW1 = RGB888 / PRW2 = RGBA8888)
    float centerX = 0.0f;      // world meters, +X right
    float centerY = 0.0f;      // world meters, +Y up
    float distance = 2.0f;     // world meters along -Z from the viewer
    float height = 0.5f;       // quad height in meters; width keeps the texture aspect
    float yawDegrees = 0.0f;   // rotation about the quad's vertical axis
    bool visible = true;
};

struct PrismSceneSettings {
    bool enabled = false;
    bool spatialWindowEnabled = false;
    // Legacy debug selector. A non-empty value restricts WindowManager tagging to one window and
    // is accepted as a fallback if metadata has not propagated yet. Empty means every eligible
    // WindowManager-tagged application surface may be spatialized; no largest-buffer fallback is
    // used.
    std::string spatialWindowLayer;
    // Default placement for newly discovered windows that have no entry in spatialWindows.
    float spatialWindowDistanceMeters = 2.0f;
    float spatialWindowHeightMeters = 0.9f;
    bool spatialWindowSuppressFlatCopy = true;
    float spatialWindowSpacingMeters = 0.20f;
    // How long the renderer remembers the automatic slot of a window whose layers left the
    // screen, so switching back to an app restores it to the same world position instead of
    // reshuffling every window. Expressed in scene frames (~display refresh).
    uint32_t spatialWindowPlacementRetentionFrames = 36000;

    // Test environment: equirect sky, RGBA menu strip, and world-space textured quads.
    // Paths are owned by SurfaceFlinger (scene file / vrender commands / property defaults)
    // and passed here by value every frame; RenderEngine caches the loaded textures by path.
    bool envEnabled = false;
    std::string skyPath;
    std::string menuPath;
    std::vector<PrismEnvQuad> envQuads;

    std::vector<PrismSpatialWindowSettings> spatialWindows;
    float ipd = 0.064f;
    float fovDegrees = 90.0f;
    // Horizontal FOV of one eye view, computed by SurfaceFlinger from the panel aspect.
    // automaticLayout clamps side-window positions into this frustum; 0 disables clamping.
    float horizontalFovDegrees = 0.0f;
    float nearPlane = 0.05f;
    float farPlane = 100.0f;
    // Capture mode (screencap, scrcpy, screen recording): render a single centered view
    // filling the whole buffer instead of the stereo pair — what the wearer sees.
    bool monoView = false;
    // Rigid transform from head space to world space. GLES uses its inverse as the view matrix;
    // the per-eye IPD transform remains separate. Debug rotations use column vectors and
    // H = R_yaw * R_pitch * R_roll about +Y, +X, and +Z respectively.
    mat4 headTransform = mat4();
};

} // namespace android::renderengine