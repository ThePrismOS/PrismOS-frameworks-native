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

#undef LOG_TAG
#define LOG_TAG "PrismSpatialLayoutTest"

#include <renderengine/PrismSpatialLayout.h>

#include <gtest/gtest.h>

namespace android::renderengine {

namespace {

constexpr float kPortraitBufferWidth = 1080.0f;
constexpr float kPortraitBufferHeight = 2248.0f;

PrismSceneSettings defaultScene() {
    PrismSceneSettings scene;
    scene.spatialWindowEnabled = true;
    return scene;
}

} // namespace

TEST(PrismStableWindowId, stripsTrailingNumericSequence) {
    EXPECT_EQ("Settings", prism::stableWindowId("Settings#12"));
    EXPECT_EQ("Settings", prism::stableWindowId("Settings#0"));
}

TEST(PrismStableWindowId, keepsNameWithoutSequenceSuffix) {
    EXPECT_EQ("Settings", prism::stableWindowId("Settings"));
    // A non-numeric trailing token is part of the title, not an SF sequence.
    EXPECT_EQ("Notes#draft", prism::stableWindowId("Notes#draft"));
}

TEST(PrismStableWindowId, stripsOnlyTheLastSegment) {
    EXPECT_EQ("My#App", prism::stableWindowId("My#App#7"));
}

TEST(PrismStableWindowId, handlesBareAndTrailingHash) {
    // A bare numeric suffix has no stable prefix: the id is the empty string, which legacy
    // untitled windows share (placement for those falls back to one shared slot).
    EXPECT_EQ("", prism::stableWindowId("#4"));
    EXPECT_EQ("Window#", prism::stableWindowId("Window#"));
    EXPECT_EQ("", prism::stableWindowId(""));
}

TEST(PrismSpatialStateKey, prefersWindowManagerId) {
    EXPECT_EQ("a123456", prism::spatialStateKey(123456, "Settings#12"));
    EXPECT_EQ("a-1", prism::spatialStateKey(-1, "Settings#12"));
}

TEST(PrismSpatialStateKey, fallsBackToStableNamePrefix) {
    EXPECT_EQ("Settings", prism::spatialStateKey(0, "Settings#12"));
    EXPECT_EQ("Settings", prism::spatialStateKey(0, "Settings"));
}

TEST(PrismAutomaticLayout, slotZeroIsCentered) {
    const PrismSceneSettings scene = defaultScene();
    const PrismSpatialWindowSettings settings =
            prism::automaticLayout(0, kPortraitBufferWidth, kPortraitBufferHeight, scene);

    EXPECT_FLOAT_EQ(0.0f, settings.centerXMeters);
    EXPECT_FLOAT_EQ(0.0f, settings.centerYMeters);
    EXPECT_FLOAT_EQ(scene.spatialWindowDistanceMeters, settings.distanceMeters);
    EXPECT_FLOAT_EQ(scene.spatialWindowHeightMeters, settings.heightMeters);
    EXPECT_EQ(scene.spatialWindowSuppressFlatCopy, settings.suppressFlatCopy);
}

TEST(PrismAutomaticLayout, firstSideSlotsAlternateAroundTheCenter) {
    const PrismSceneSettings scene = defaultScene();
    const float width =
            scene.spatialWindowHeightMeters * kPortraitBufferWidth / kPortraitBufferHeight;
    const float pitch = width + scene.spatialWindowSpacingMeters;

    const PrismSpatialWindowSettings first =
            prism::automaticLayout(1, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_FLOAT_EQ(-pitch, first.centerXMeters);
    EXPECT_FLOAT_EQ(0.0f, first.centerYMeters);

    const PrismSpatialWindowSettings second =
            prism::automaticLayout(2, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_FLOAT_EQ(pitch, second.centerXMeters);
    EXPECT_FLOAT_EQ(0.0f, second.centerYMeters);

    const PrismSpatialWindowSettings third =
            prism::automaticLayout(3, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_FLOAT_EQ(-2.0f * pitch, third.centerXMeters);
}

TEST(PrismAutomaticLayout, wrapsIntoLowerRowAfterFourSideSlots) {
    const PrismSceneSettings scene = defaultScene();
    const float width =
            scene.spatialWindowHeightMeters * kPortraitBufferWidth / kPortraitBufferHeight;
    const float pitch = width + scene.spatialWindowSpacingMeters;

    // Slot 4 is still in the first row on the right; slot 5 wraps below, back at the inner ring.
    const PrismSpatialWindowSettings fourth =
            prism::automaticLayout(4, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_FLOAT_EQ(2.0f * pitch, fourth.centerXMeters);
    EXPECT_FLOAT_EQ(0.0f, fourth.centerYMeters);

    const PrismSpatialWindowSettings fifth =
            prism::automaticLayout(5, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_FLOAT_EQ(-3.0f * pitch, fifth.centerXMeters);
    EXPECT_FLOAT_EQ(-(scene.spatialWindowHeightMeters + scene.spatialWindowSpacingMeters),
                    fifth.centerYMeters);
}

TEST(PrismAutomaticLayout, keepsBufferAspectRatio) {
    const PrismSceneSettings scene = defaultScene();
    const PrismSpatialWindowSettings landscape =
            prism::automaticLayout(0, 2248.0f, 1080.0f, scene);
    // The quad dims are not part of the settings; the aspect shows up through the ring pitch of
    // a window sharing the scene with it, so verify via slot 1 which uses the same buffer and
    // sits on the left of the center slot.
    const PrismSpatialWindowSettings side =
            prism::automaticLayout(1, 2248.0f, 1080.0f, scene);
    const float landscapeWidth = scene.spatialWindowHeightMeters * 2248.0f / 1080.0f;
    EXPECT_FLOAT_EQ(-(landscapeWidth + scene.spatialWindowSpacingMeters), side.centerXMeters);
    EXPECT_LT(side.centerXMeters, 0.0f);
    EXPECT_GT(landscape.distanceMeters, 0.0f);
}

TEST(PrismAutomaticLayout, clampsDegenerateSceneValues) {
    PrismSceneSettings scene = defaultScene();
    scene.spatialWindowHeightMeters = 0.0f;
    scene.spatialWindowDistanceMeters = 0.0f;
    const PrismSpatialWindowSettings settings =
            prism::automaticLayout(1, kPortraitBufferWidth, kPortraitBufferHeight, scene);
    EXPECT_GE(settings.heightMeters, 0.05f);
    EXPECT_GE(settings.distanceMeters, 0.25f);
}

} // namespace android::renderengine
