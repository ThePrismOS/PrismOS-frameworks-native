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


#include "PrismHeadTracker.h"

#include <android-base/properties.h>
#include <android/sensor.h>
#include <binder/IServiceManager.h>
#include <log/log.h>
#include <sensor/Sensor.h>
#include <sensor/SensorEventQueue.h>
#include <sensor/SensorManager.h>
#include <utils/Looper.h>
#include <utils/String16.h>
#include <utils/String8.h>
#include <utils/Timers.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>

namespace android {
namespace {

constexpr char kSensorServiceName[] = "sensorservice";
constexpr char kClientName[] = "SurfaceFlinger";
constexpr int32_t kSamplingPeriodUs = 20'000;
constexpr int kSensorFdIdent = 1;
// PrismOS: poll timeout + staleness watchdog. SensorService can silently stop delivering on
// a connection it keeps open (a display off/on cycle does this); a blocking pollOnce would
// then hang the tracker forever with no events and no hangup. Poll on a timeout and rebuild
// the queue when no sample arrives for a while, so the head pipeline self-heals.
constexpr int kSensorPollTimeoutMs = 1000;
constexpr int64_t kSensorStaleNs = 3'000'000'000ll; // 3 s without a sample → reconnect
constexpr auto kReconnectDelay = std::chrono::seconds(1);
constexpr int kSnapshotAttempts = 3;
constexpr double kMinimumQuaternionNormSquared = 1.0e-12;

// PrismOS 3-DOF processing. The rotation-vector sensor's identity orientation corresponds to
// the device lying flat on a surface, not to the wearer's forward gaze in a viewer. The
// pipeline applies a configurable mounting correction, recenters onto the first valid sample
// (or on demand), and nlerp-filters with a small deadband before publishing:
//   debug.prismos.head.mount_yaw / mount_pitch / mount_roll  (degrees, pitch default -90)
//   debug.prismos.head.recenter                              (1 = capture zero, auto-clears)
//   debug.prismos.head.invert                                (1 = invert relative rotation)
//   debug.prismos.head.smoothing                             (0..0.95, default 0.4)
//   debug.prismos.head.deadband_deg                          (default 0.2)
constexpr char kMountYawKey[] = "debug.prismos.head.mount_yaw";
constexpr char kMountPitchKey[] = "debug.prismos.head.mount_pitch";
constexpr char kMountRollKey[] = "debug.prismos.head.mount_roll";
constexpr char kRecenterKey[] = "debug.prismos.head.recenter";
constexpr char kInvertKey[] = "debug.prismos.head.invert";
constexpr char kSmoothingKey[] = "debug.prismos.head.smoothing";
constexpr char kDeadbandKey[] = "debug.prismos.head.deadband_deg";
constexpr float kDefaultMountPitchDegrees = -90.0f;
constexpr float kDefaultSmoothing = 0.4f;
constexpr float kDefaultDeadbandDegrees = 0.2f;

struct PrismQuat {
    float w;
    float x;
    float y;
    float z;
};

float prismFloatProperty(const char* key, float defaultValue) {
    const std::string value = android::base::GetProperty(key, "");
    if (value.empty()) {
        return defaultValue;
    }
    const float parsed = strtof(value.c_str(), nullptr);
    return std::isfinite(parsed) ? parsed : defaultValue;
}

PrismQuat quatMultiply(const PrismQuat& a, const PrismQuat& b) {
    PrismQuat result;
    result.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    result.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    result.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    result.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return result;
}

PrismQuat quatConjugate(const PrismQuat& q) {
    return {q.w, -q.x, -q.y, -q.z};
}

PrismQuat quatNormalized(const PrismQuat& q) {
    const double normSquared = static_cast<double>(q.w) * q.w + static_cast<double>(q.x) * q.x +
            static_cast<double>(q.y) * q.y + static_cast<double>(q.z) * q.z;
    if (normSquared < kMinimumQuaternionNormSquared) {
        return {1.0f, 0.0f, 0.0f, 0.0f};
    }
    const double inverseNorm = 1.0 / std::sqrt(normSquared);
    return {static_cast<float>(q.w * inverseNorm), static_cast<float>(q.x * inverseNorm),
            static_cast<float>(q.y * inverseNorm), static_cast<float>(q.z * inverseNorm)};
}

float quatDot(const PrismQuat& a, const PrismQuat& b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

PrismQuat quatFromAxisAngleDegrees(const float axis[3], float degrees) {
    const float radians = degrees * static_cast<float>(M_PI) / 180.0f;
    const float half = radians * 0.5f;
    const float s = std::sin(half);
    return {std::cos(half), axis[0] * s, axis[1] * s, axis[2] * s};
}

PrismQuat quatFromMountProperties() {
    const float yaw = prismFloatProperty(kMountYawKey, 0.0f);
    const float pitch = prismFloatProperty(kMountPitchKey, kDefaultMountPitchDegrees);
    const float roll = prismFloatProperty(kMountRollKey, 0.0f);

    const float yAxis[3] = {0.0f, 1.0f, 0.0f};
    const float xAxis[3] = {1.0f, 0.0f, 0.0f};
    const float zAxis[3] = {0.0f, 0.0f, 1.0f};
    return quatNormalized(quatMultiply(
            quatMultiply(quatFromAxisAngleDegrees(yAxis, yaw),
                         quatFromAxisAngleDegrees(xAxis, pitch)),
            quatFromAxisAngleDegrees(zAxis, roll)));
}

uint32_t floatToBits(float value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bitsToFloat(uint32_t bits) {
    float value;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

PrismHeadTracker::~PrismHeadTracker() {
    stop();
}

void PrismHeadTracker::setInvalidateCallback(std::function<void()> callback) {
    mInvalidateCallback = std::move(callback);
}

void PrismHeadTracker::start() {
    bool expected = false;
    if (!mStarted.compare_exchange_strong(expected, true)) {
        return;
    }

    mStopRequested.store(false, std::memory_order_release);
    mThread = std::thread(&PrismHeadTracker::threadMain, this);
}

void PrismHeadTracker::stop() {
    if (!mStarted.exchange(false)) {
        return;
    }

    mStopRequested.store(true, std::memory_order_release);
    mStopCondition.notify_all();

    sp<Looper> looper;
    {
        std::lock_guard lock(mLooperMutex);
        looper = mLooper;
    }
    if (looper != nullptr) {
        looper->wake();
    }

    if (mThread.joinable()) {
        mThread.join();
    }
    invalidate();
}

bool PrismHeadTracker::getLatestQuaternion(Quaternion* quaternion) const {
    if (quaternion == nullptr) {
        return false;
    }

    for (int attempt = 0; attempt < kSnapshotAttempts; ++attempt) {
        const uint32_t sequenceBefore = mSequence.load(std::memory_order_acquire);
        if ((sequenceBefore & 1U) != 0U) {
            continue;
        }

        const bool valid = mPoseValid.load(std::memory_order_relaxed);
        const Quaternion snapshot{
                .x = bitsToFloat(mQuaternionX.load(std::memory_order_relaxed)),
                .y = bitsToFloat(mQuaternionY.load(std::memory_order_relaxed)),
                .z = bitsToFloat(mQuaternionZ.load(std::memory_order_relaxed)),
                .w = bitsToFloat(mQuaternionW.load(std::memory_order_relaxed)),
        };

        const uint32_t sequenceAfter = mSequence.load(std::memory_order_acquire);
        if (sequenceBefore == sequenceAfter && (sequenceAfter & 1U) == 0U) {
            if (!valid) {
                return false;
            }
            *quaternion = snapshot;
            return true;
        }
    }

    return false;
}

void PrismHeadTracker::publishQuaternion(float x, float y, float z, float w) {
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w)) {
        return;
    }

    const double normSquared = static_cast<double>(x) * x + static_cast<double>(y) * y +
            static_cast<double>(z) * z + static_cast<double>(w) * w;
    if (!std::isfinite(normSquared) || normSquared < kMinimumQuaternionNormSquared) {
        return;
    }

    const double inverseNorm = 1.0 / std::sqrt(normSquared);
    x = static_cast<float>(static_cast<double>(x) * inverseNorm);
    y = static_cast<float>(static_cast<double>(y) * inverseNorm);
    z = static_cast<float>(static_cast<double>(z) * inverseNorm);
    w = static_cast<float>(static_cast<double>(w) * inverseNorm);
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w)) {
        return;
    }

    mSequence.fetch_add(1, std::memory_order_acq_rel);
    mQuaternionX.store(floatToBits(x), std::memory_order_relaxed);
    mQuaternionY.store(floatToBits(y), std::memory_order_relaxed);
    mQuaternionZ.store(floatToBits(z), std::memory_order_relaxed);
    mQuaternionW.store(floatToBits(w), std::memory_order_relaxed);
    mPoseValid.store(true, std::memory_order_relaxed);
    mSequence.fetch_add(1, std::memory_order_release);
}

// PrismOS raw-sensor fusion: the gyro integrates the orientation (fast, drifts), the
// accelerometer pulls the tilt back to the gravity direction (slow, stable). The gyro bias
// adapts automatically whenever the device rests still for a couple of seconds, which resets
// the yaw drift without any user action.
void PrismHeadTracker::onGyroEvent(float wx, float wy, float wz, int64_t timestampNs) {
    // Static bias adaptation: while the device is nearly still, slowly absorb the bias.
    const float gyroMagnitude = std::sqrt(wx * wx + wy * wy + wz * wz);
    if (gyroMagnitude < 0.04f) {
        mGyroBiasX += (wx - mGyroBiasX) * 0.01f;
        mGyroBiasY += (wy - mGyroBiasY) * 0.01f;
        mGyroBiasZ += (wz - mGyroBiasZ) * 0.01f;
    }
    const float wxC = wx - mGyroBiasX;
    const float wyC = wy - mGyroBiasY;
    const float wzC = wz - mGyroBiasZ;

    if (mLastGyroTimestamp == 0) {
        mLastGyroTimestamp = timestampNs;
        return;
    }
    const float dt = static_cast<float>(static_cast<double>(timestampNs - mLastGyroTimestamp) *
                                        1e-9);
    mLastGyroTimestamp = timestampNs;
    if (dt <= 0.0f || dt > 0.5f) {
        return;
    }

    const float omegaX = wxC * dt;
    const float omegaY = wyC * dt;
    const float omegaZ = wzC * dt;
    const float angle = std::sqrt(omegaX * omegaX + omegaY * omegaY + omegaZ * omegaZ);
    if (angle > 1e-7f) {
        const float half = angle * 0.5f;
        const float s = std::sin(half) / angle;
        const PrismQuat delta{std::cos(half), omegaX * s, omegaY * s, omegaZ * s};
        // Device-frame rotation pre-multiplies: world = delta ⊗ fused.
        const PrismQuat fused{mFused.w, mFused.x, mFused.y, mFused.z};
        const PrismQuat multiplied = quatMultiply(delta, fused);
        const PrismQuat normalized = quatNormalized(multiplied);
        mFused = {normalized.x, normalized.y, normalized.z, normalized.w};
    }
}

void PrismHeadTracker::onAccelEvent(float ax, float ay, float az) {
    const float magnitude = std::sqrt(ax * ax + ay * ay + az * az);
    if (magnitude < 5.0f || magnitude > 15.0f) {
        // Not a trustworthy gravity reading (free fall / heavy shake).
        return;
    }

    // The measured gravity direction (the device frame): the accelerometer reads the reaction
    // force (+up when flat), so gravity points opposite the reading.
    const float invMagnitude = 1.0f / magnitude;
    const float mx = -ax * invMagnitude;
    const float my = -ay * invMagnitude;
    const float mz = -az * invMagnitude;

    // The gravity direction in the device frame implied by the current fused orientation:
    // the world down vector (0,0,-1) rotated by the INVERSE of the device-to-world quaternion
    // (world -> device). For q = (w,x,y,z) the rotated (0,0,-1) is
    // (-2(xz-wy), -2(yz+wx), 2(x^2+y^2)-1).
    const float gx = -2.0f * (mFused.x * mFused.z - mFused.w * mFused.y);
    const float gy = -2.0f * (mFused.y * mFused.z + mFused.w * mFused.x);
    const float gz = 2.0f * (mFused.x * mFused.x + mFused.y * mFused.y) - 1.0f;

    // Tilt error: the rotation that would bring the estimated gravity onto the measured one.
    const float crossX = gy * mz - gz * my;
    const float crossY = gz * mx - gx * mz;
    const float crossZ = gx * my - gy * mx;
    const float crossNorm = std::sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
    if (crossNorm < 1e-5f) {
        mAccelSeen = true;
        return;
    }
    const float cosAngle = std::clamp(gx * mx + gy * my + gz * mz, -1.0f, 1.0f);
    const float errorAngle = std::acos(cosAngle);
    if (errorAngle < 0.001f) {
        mAccelSeen = true;
        return;
    }

    // The complementary correction: a small step (2%) around the error axis, in the world
    // frame (pre-multiplied).
    const float factor = 0.02f;
    const float axisX = crossX / crossNorm;
    const float axisY = crossY / crossNorm;
    const float axisZ = crossZ / crossNorm;
    const float half = errorAngle * factor * 0.5f;
    const float s = std::sin(half);
    const PrismQuat correction{std::cos(half), axisX * s, axisY * s, axisZ * s};
    const PrismQuat fused{mFused.w, mFused.x, mFused.y, mFused.z};
    const PrismQuat corrected = quatNormalized(quatMultiply(correction, fused));
    mFused = {corrected.x, corrected.y, corrected.z, corrected.w};
    mAccelSeen = true;
}

// Mount-corrects the fused orientation, recenters it onto the wearer's forward gaze, and
// nlerp-filters the result before publishing. Runs on the sensor thread only.
void PrismHeadTracker::processAndPublish(float x, float y, float z, float w) {
    const PrismQuat sensor{w, x, y, z};

    // 1. Mounting correction: sensor identity is "device flat", the viewer mounts it rotated.
    const PrismQuat mounted = quatMultiply(quatFromMountProperties(), sensor);

    // 2. Re-zero: capture the mounted orientation as the wearer's forward gaze. Automatic on
    // the first valid sample after start/reconnect, manual via the recenter property.
    if (android::base::GetProperty(kRecenterKey, "0") == "1") {
        mReferenceValid = false;
        android::base::SetProperty(kRecenterKey, "0");
    }
    if (!mReferenceValid) {
        mReference = Quaternion{mounted.x, mounted.y, mounted.z, mounted.w};
        mReferenceValid = true;
        mSmoothedValid = false;
        ALOGI("PrismHeadTracker: recentered, zero point captured");
    }
    const PrismQuat reference{mReference.w, mReference.x, mReference.y, mReference.z};

    // 3. Relative orientation: head motion away from the captured forward gaze.
    PrismQuat relative = quatMultiply(quatConjugate(reference), mounted);
    if (android::base::GetBoolProperty(kInvertKey, false)) {
        relative = quatConjugate(relative);
    }
    const PrismQuat target = quatNormalized(relative);

    // 4. nlerp smoothing with a small deadband to suppress sensor micro-jitter.
    const float smoothing = std::clamp(
            prismFloatProperty(kSmoothingKey, kDefaultSmoothing), 0.0f, 0.95f);
    const float deadbandDeg = prismFloatProperty(kDeadbandKey, kDefaultDeadbandDegrees);

    PrismQuat smoothed{1.0f, 0.0f, 0.0f, 0.0f};
    if (!mSmoothedValid) {
        smoothed = target;
        mSmoothedValid = true;
    } else {
        const PrismQuat previous{mSmoothed.w, mSmoothed.x, mSmoothed.y, mSmoothed.z};
        PrismQuat blended = target;
        float dot = quatDot(previous, blended);
        if (dot < 0.0f) {
            blended = {-blended.w, -blended.x, -blended.y, -blended.z};
            dot = -dot;
        }
        const float angleDeg = 2.0f * std::acos(std::clamp(dot, -1.0f, 1.0f)) *
                180.0f / static_cast<float>(M_PI);
        if (angleDeg >= deadbandDeg) {
            smoothed = quatNormalized({previous.w + smoothing * (blended.w - previous.w),
                                       previous.x + smoothing * (blended.x - previous.x),
                                       previous.y + smoothing * (blended.y - previous.y),
                                       previous.z + smoothing * (blended.z - previous.z)});
        } else {
            smoothed = previous;
        }
    }
    mSmoothed = Quaternion{smoothed.x, smoothed.y, smoothed.z, smoothed.w};

    // Head-rotation invalidation: schedule a composition when the orientation moved beyond a
    // small deadband. The layer content does not change with the head, so without this
    // SurfaceFlinger would not compose new frames and the scene would never track.
    if (mInvalidateCallback) {
        const PrismQuat last{mLastPublished.w, mLastPublished.x, mLastPublished.y,
                             mLastPublished.z};
        const PrismQuat current{mSmoothed.w, mSmoothed.x, mSmoothed.y, mSmoothed.z};
        const float dot = quatDot(last, current);
        const float angleDeg = 2.0f * std::acos(std::clamp(dot, -1.0f, 1.0f)) * 180.0f /
                static_cast<float>(M_PI);
        if (angleDeg > 0.1f) {
            mLastPublished = mSmoothed;
            mInvalidateCallback();
        }
    }

    publishQuaternion(mSmoothed.x, mSmoothed.y, mSmoothed.z, mSmoothed.w);
}

void PrismHeadTracker::invalidate() {
    mSequence.fetch_add(1, std::memory_order_acq_rel);
    mPoseValid.store(false, std::memory_order_relaxed);
    mSequence.fetch_add(1, std::memory_order_release);
    // Drop the smoothing reference so the next valid sample recenters onto the wearer's
    // current forward gaze instead of snapping back to a stale zero point. Benign race with
    // the sensor thread: the worst case is one extra recenter.
    mReferenceValid = false;
    mSmoothedValid = false;
}

void PrismHeadTracker::setLooper(const sp<Looper>& looper) {
    std::lock_guard lock(mLooperMutex);
    mLooper = looper;
}

void PrismHeadTracker::clearLooper(const sp<Looper>& looper) {
    std::lock_guard lock(mLooperMutex);
    if (mLooper == looper) {
        mLooper.clear();
    }
}

bool PrismHeadTracker::waitForReconnect() {
    std::unique_lock lock(mStopMutex);
    mStopCondition.wait_for(lock, kReconnectDelay, [this] {
        return mStopRequested.load(std::memory_order_acquire);
    });
    return !mStopRequested.load(std::memory_order_acquire);
}

void PrismHeadTracker::threadMain() {
    while (!mStopRequested.load(std::memory_order_acquire)) {
        const sp<IServiceManager> serviceManager = defaultServiceManager();
        if (serviceManager == nullptr ||
            serviceManager->checkService(String16(kSensorServiceName)) == nullptr) {
            invalidate();
            if (!waitForReconnect()) {
                break;
            }
            continue;
        }

        SensorManager& sensorManager =
                SensorManager::getInstanceForPackage(String16(kClientName));
        // PrismOS: raw accelerometer + gyroscope with an in-tracker complementary filter.
        // The fused rotation-vector sensors on this device report a frozen identity, while
        // the raw sensors always deliver.
        const Sensor* accelSensor = sensorManager.getDefaultSensor(ASENSOR_TYPE_ACCELEROMETER);
        const Sensor* gyroSensor = sensorManager.getDefaultSensor(ASENSOR_TYPE_GYROSCOPE);
        if (accelSensor == nullptr || gyroSensor == nullptr) {
            invalidate();
            if (!waitForReconnect()) {
                break;
            }
            continue;
        }

        sp<SensorEventQueue> queue =
                sensorManager.createEventQueue(String8(kClientName));
        if (queue == nullptr) {
            //ALOGW("Failed to create sensor event queue");
            invalidate();
            if (!waitForReconnect()) {
                break;
            }
            continue;
        }

        const status_t enableStatus = queue->enableSensor(accelSensor, kSamplingPeriodUs);
        const status_t enableGyroStatus =
                queue->enableSensor(gyroSensor, kSamplingPeriodUs);
        if (enableStatus != NO_ERROR || enableGyroStatus != NO_ERROR) {
            //ALOGW("Failed to enable rotation vector sensor: %d", enableStatus);
            invalidate();
            if (!waitForReconnect()) {
                break;
            }
            continue;
        }

        sp<Looper> looper = new Looper(true);
        const int sensorFd = queue->getFd();

        //ALOGI("PrismHeadTracker: sensor=%d queue=%p fd=%d", sensorType, queue.get(), sensorFd);
        
        errno = 0;
        
        const int addFdStatus =
                looper->addFd(sensorFd, kSensorFdIdent,
                              ALOOPER_EVENT_INPUT | ALOOPER_EVENT_ERROR |
                                      ALOOPER_EVENT_HANGUP,
                              nullptr, nullptr);
                
        //ALOGI("PrismHeadTracker: addFd(fd=%d) -> %d errno=%d (%s)", sensorFd, addFdStatus, errno, strerror(errno));
                
        if (addFdStatus < 0) {
            //ALOGW("Failed to add sensor queue fd to looper: %d", addFdStatus);
            queue->disableSensor(accelSensor);
            queue->disableSensor(gyroSensor);
            invalidate();
            if (!waitForReconnect()) {
                break;
            }
            continue;
        }

        setLooper(looper);
        bool reconnect = false;
        int64_t lastEventNsecs = systemTime();
        while (!mStopRequested.load(std::memory_order_acquire) && !reconnect) {
            int events = 0;
            const int pollResult =
                    looper->pollOnce(kSensorPollTimeoutMs, nullptr, &events, nullptr);
            if (pollResult == ALOOPER_POLL_TIMEOUT) {
                // No fd activity within the window: if the sensor stream has gone silent
                // (connection open but suspended), rebuild it instead of waiting forever.
                if (systemTime() - lastEventNsecs > kSensorStaleNs) {
                    ALOGW("PrismHeadTracker: no sensor samples for %.1f s; reconnecting",
                          static_cast<double>(systemTime() - lastEventNsecs) / 1e9);
                    reconnect = true;
                }
                continue;
            }
            if (pollResult == ALOOPER_POLL_WAKE) {
                continue;
            }
            if (pollResult == ALOOPER_POLL_ERROR ||
                (pollResult == kSensorFdIdent &&
                 (events & (ALOOPER_EVENT_ERROR | ALOOPER_EVENT_HANGUP)) != 0)) {
                reconnect = true;
                break;
            }
            if (pollResult != kSensorFdIdent || (events & ALOOPER_EVENT_INPUT) == 0) {
                continue;
            }

            std::array<ASensorEvent, 16> sensorEvents;
            for (;;) {
                const ssize_t count = queue->read(sensorEvents.data(), sensorEvents.size());
    if (count < 0) {
        if (count != -EAGAIN) {
            reconnect = true;
        }
        break;
    }

    if (count == 0) {
        break;
    }

    lastEventNsecs = systemTime();

    for (ssize_t i = 0; i < count; ++i) {
        const ASensorEvent& event =
                sensorEvents[static_cast<size_t>(i)];

        /*ALOGI("PrismHeadTracker: event type=%d quat=[%.4f %.4f %.4f %.4f]",
              event.type,
              event.data[0],
              event.data[1],
              event.data[2],
              event.data[3]);*/

        if (event.type == ASENSOR_TYPE_GYROSCOPE) {
            onGyroEvent(event.data[0], event.data[1], event.data[2], event.timestamp);
            // Publish at the gyro rate: the fusion updated mFused, run the head pipeline.
            processAndPublish(mFused.x, mFused.y, mFused.z, mFused.w);
        } else if (event.type == ASENSOR_TYPE_ACCELEROMETER) {
            onAccelEvent(event.data[0], event.data[1], event.data[2]);
        }
    }
                
                queue->sendAck(sensorEvents.data(), static_cast<int>(count));
            }
        }

        clearLooper(looper);
        looper->removeFd(queue->getFd());
        queue->disableSensor(accelSensor);
            queue->disableSensor(gyroSensor);
        invalidate();

        if (!mStopRequested.load(std::memory_order_acquire) && !waitForReconnect()) {
            break;
        }
    }

    invalidate();
}

// ---------------------------------------------------------------------------
// PrismHandTracker (stub)
//
// The runtime skeleton for 6-DOF hand tracking: property-gated lifecycle, per-hand state
// storage with thread-safe publication, and an access API for the future scene renderer.
// The capture source (camera HAL feed, neural-network service, or controller input) is not
// wired yet; until something calls publishHandPose(), every pose reads back as invalid.
//   debug.prismos.hands.enabled  (1 = tracker starts alongside the head tracker)
// ---------------------------------------------------------------------------

void PrismHandTracker::start() {
    if (!android::base::GetBoolProperty("debug.prismos.hands.enabled", false)) {
        ALOGI("PrismHandTracker: disabled (debug.prismos.hands.enabled not set)");
        return;
    }
    if (mStarted.exchange(true)) {
        return;
    }
    ALOGI("PrismHandTracker: started (stub, awaiting capture source)");
}

void PrismHandTracker::stop() {
    mStarted.store(false);
    for (auto& hand : mHands) {
        std::lock_guard lock(hand.mutex);
        hand.pose.valid = false;
    }
}

void PrismHandTracker::publishHandPose(int handId, const HandPose& pose) {
    if (handId < 0 || handId >= kMaxHands) {
        return;
    }
    auto& state = mHands[static_cast<size_t>(handId)];
    std::lock_guard lock(state.mutex);
    state.pose = pose;
}

bool PrismHandTracker::getLatestHandPose(int handId, HandPose* out) const {
    if (handId < 0 || handId >= kMaxHands || out == nullptr) {
        return false;
    }
    auto& state = mHands[static_cast<size_t>(handId)];
    std::lock_guard lock(state.mutex);
    *out = state.pose;
    return state.pose.valid;
}

} // namespace android