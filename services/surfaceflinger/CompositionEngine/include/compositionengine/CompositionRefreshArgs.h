/*
 * Copyright 2019 The Android Open Source Project
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

#include <chrono>
#include <optional>
#include <vector>

#include <compositionengine/Display.h>
#include <compositionengine/LayerFE.h>
#include <compositionengine/OutputColorSetting.h>
#include <math/mat4.h>
#include <renderengine/PrismScene.h>
#include <ui/DisplayIdentification.h>
#include <ui/FenceTime.h>
#include <ui/Transform.h>

namespace android::compositionengine {

using Layers = std::vector<sp<compositionengine::LayerFE>>;
using Outputs = std::vector<std::shared_ptr<compositionengine::Output>>;

/**
 * Describes a stereo composition performed by one Output and one target.
 *
 * The plan is deliberately expressed in terms of per-eye transforms rather
 * than Android displays. Output expands its client-composition requests into
 * two views of the same LayerFE set and packs them into the existing target.
 */
struct StereoCompositionPlan {
    enum class Layout {
        kSideBySide,
    };

    bool enabled{false};
    Layout layout{Layout::kSideBySide};

    // Identity is the initial PrismOS view. These matrices are in the logical
    // layer-stack space and can later be replaced by head-pose and eye-offset
    // transforms without changing the WindowManager hierarchy.
    mat4 leftViewTransform{};
    mat4 rightViewTransform{};

    // Restrict activation to one existing logical display. An unset value is
    // useful for tests and means that the plan applies to every output.
    std::optional<DisplayId> displayId;
};

/**
 * A parameter object for refreshing a set of outputs
 */
struct CompositionRefreshArgs {
    // All the outputs being refreshed
    Outputs outputs;

    // Optional single-output stereo composition. Empty for the existing mono
    // Android path. The target remains the RenderSurface owned by that output.
    std::optional<StereoCompositionPlan> stereoComposition;

    // Optional GLES-only bring-up scene. The target remains the same client target owned by the
    // Output; this does not create another display or RenderSurface.
    std::optional<renderengine::PrismSceneSettings> prismScene;

    // All the layers that are potentially visible in the outputs. The order of
    // the layers is important, and should be in traversal order from back to
    // front.
    Layers layers;

    // All the layers that have queued updates.
    Layers layersWithQueuedFrames;

    // Controls how the color mode is chosen for an output
    OutputColorSetting outputColorSetting{OutputColorSetting::kEnhanced};

    // If not Dataspace::UNKNOWN, overrides the dataspace on each output
    ui::Dataspace colorSpaceAgnosticDataspace{ui::Dataspace::UNKNOWN};

    // Forces a color mode on the outputs being refreshed
    ui::ColorMode forceOutputColorMode{ui::ColorMode::NATIVE};

    // Used to correctly apply an inverse-display buffer transform if applicable
    ui::Transform::RotationFlags internalDisplayRotationFlags{ui::Transform::ROT_0};

    // If true, GPU clocks will be increased when rendering blurs
    bool blursAreExpensive{false};

    // If true, the complete output geometry needs to be recomputed this frame
    bool updatingOutputGeometryThisFrame{false};

    // If true, there was a geometry update this frame
    bool updatingGeometryThisFrame{false};

    // The color matrix to use for this
    // frame. Only set if the color transform is changing this frame.
    std::optional<mat4> colorTransformMatrix;

    // If true, client composition is always used.
    bool devOptForceClientComposition{false};

    // If set, causes the dirty regions to flash with the delay
    std::optional<std::chrono::microseconds> devOptFlashDirtyRegionsDelay;

    // The earliest time to send the present command to the HAL
    std::chrono::steady_clock::time_point earliestPresentTime;

    // The previous present fence. Used together with earliestPresentTime
    // to prevent an early presentation of a frame.
    std::shared_ptr<FenceTime> previousPresentFence;

    // The expected time for the next present
    nsecs_t expectedPresentTime{0};

    // If set, a frame has been scheduled for that time.
    std::optional<std::chrono::steady_clock::time_point> scheduledFrameTime;
};

} // namespace android::compositionengine
