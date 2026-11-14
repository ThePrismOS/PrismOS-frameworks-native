/*
 * Copyright (C) 2026 The Android Open Source Project
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

#include <utils/StrongPointer.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>

namespace android {

class Looper;

// 3-DOF head orientation tracker. Raw rotation-vector sensor data is processed on the sensor
// thread before being published: a configurable mounting correction turns the sensor's
// flat-on-the-floor reference frame into the viewer's portrait mounting, the first valid
// sample (or a manual recenter) becomes the zero point so the scene forward matches the
// wearer's forward gaze, and an nlerp filter with a small deadband smooths sensor jitter.
// All tuning is done through debug.prismos.head.* properties and takes effect live.
class PrismHeadTracker {
public:
    struct Quaternion {
        float x;
        float y;
        float z;
        float w;
    };

    PrismHeadTracker() = default;
    ~PrismHeadTracker();

    PrismHeadTracker(const PrismHeadTracker&) = delete;
    PrismHeadTracker& operator=(const PrismHeadTracker&) = delete;

    void start();
    void stop();

    // PrismOS: the callback invoked when the processed orientation changes beyond a small
    // deadband — SurfaceFlinger uses it to schedule a composition so the scene tracks the
    // head continuously (the layer content itself does not change, so nothing else would
    // trigger new frames). Must be set before start().
    void setInvalidateCallback(std::function<void()> callback);

    // Returns the latest processed orientation (mount-corrected, recentered, smoothed).
    bool getLatestQuaternion(Quaternion* quaternion) const;

private:
    void threadMain();
    bool waitForReconnect();
    void publishQuaternion(float x, float y, float z, float w);
    void processAndPublish(float x, float y, float z, float w);
    // Raw-sensor 3-DOF fusion (gyro integration + accelerometer gravity correction). The
    // fused rotation-vector sensors on this device report a frozen identity, so the tracker
    // integrates the gyroscope directly and drift-corrects with the accelerometer.
    void onGyroEvent(float wx, float wy, float wz, int64_t timestampNs);
    void onAccelEvent(float ax, float ay, float az);
    void invalidate();
    void setLooper(const sp<Looper>& looper);
    void clearLooper(const sp<Looper>& looper);

    std::atomic<bool> mStarted = false;
    std::atomic<bool> mStopRequested = false;
    std::thread mThread;

    std::mutex mStopMutex;
    std::condition_variable mStopCondition;

    std::mutex mLooperMutex;
    sp<Looper> mLooper;

    std::atomic<uint32_t> mSequence = 0;
    std::atomic<uint32_t> mQuaternionX = 0;
    std::atomic<uint32_t> mQuaternionY = 0;
    std::atomic<uint32_t> mQuaternionZ = 0;
    std::atomic<uint32_t> mQuaternionW = 0;
    std::atomic<bool> mPoseValid = false;

    // 3-DOF processing state. Only touched on the sensor thread between the seqlock
    // publication fences, so no additional locking is required.
    Quaternion mReference{0.0f, 0.0f, 0.0f, 1.0f};
    Quaternion mSmoothed{0.0f, 0.0f, 0.0f, 1.0f};
    bool mReferenceValid = false;
    bool mSmoothedValid = false;

    // Raw-sensor fusion state (gyro integration + accelerometer gravity correction), sensor
    // thread only. mFused is the device-to-world orientation maintained by the complementary
    // filter — the fused rotation-vector sensors on this device report a frozen identity.
    Quaternion mFused{0.0f, 0.0f, 0.0f, 1.0f};
    int64_t mLastGyroTimestamp = 0;
    float mGyroBiasX = 0.0f;
    float mGyroBiasY = 0.0f;
    float mGyroBiasZ = 0.0f;
    bool mAccelSeen = false;

    // Head-rotation invalidation: the callback and the last published orientation (both the
    // sensor thread only). The callback fires when the rotation moves beyond the deadband so
    // SurfaceFlinger schedules a composition; without it a head turn would not trigger new
    // frames at all (the layer content itself does not change).
    std::function<void()> mInvalidateCallback;
    Quaternion mLastPublished{0.0f, 0.0f, 0.0f, 1.0f};
};

// PrismOS hand-tracking stub. The runtime plumbing (state storage, thread-safe publication,
// property gating, lifecycle next to the head tracker) exists so that a capture source — a
// camera HAL feed, a neural-network service, or controller input — can be wired in later
// without touching SurfaceFlinger again. Until a source calls publishHandPose(), poses read
// back as invalid and nothing in the scene changes.
class PrismHandTracker {
public:
    struct HandPose {
        bool valid = false;
        float positionMeters[3] = {0.0f, 0.0f, 0.0f};      // world space, head-referenced
        float orientation[4] = {0.0f, 0.0f, 0.0f, 1.0f};   // quaternion x, y, z, w
        float pinchStrength = 0.0f;                        // 0..1, index-thumb pinch
    };

    static constexpr int kMaxHands = 2;  // left, right

    PrismHandTracker() = default;
    ~PrismHandTracker() { stop(); }

    PrismHandTracker(const PrismHandTracker&) = delete;
    PrismHandTracker& operator=(const PrismHandTracker&) = delete;

    // No-op unless debug.prismos.hands.enabled is set; keeps the lifecycle symmetric with the
    // head tracker so the future capture source can simply start publishing into it.
    void start();
    void stop();

    // Integration point for the future capture source.
    void publishHandPose(int handId, const HandPose& pose);

    bool getLatestHandPose(int handId, HandPose* out) const;

private:
    struct HandState {
        mutable std::mutex mutex;
        HandPose pose;
    };

    std::atomic<bool> mStarted = false;
    std::array<HandState, kMaxHands> mHands;
};

} // namespace android
