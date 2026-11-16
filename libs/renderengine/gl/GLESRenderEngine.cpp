/*
 * Copyright 2013 The Android Open Source Project
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

//#define LOG_NDEBUG 0
#include "EGL/egl.h"
#undef LOG_TAG
#define LOG_TAG "RenderEngine"
#define ATRACE_TAG ATRACE_TAG_GRAPHICS

#include <sched.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <android-base/properties.h>
#include <array>
#include <android-base/stringprintf.h>
#include <cutils/compiler.h>
#include <cutils/properties.h>
#include <gui/DebugEGLImageTracker.h>
#include <renderengine/Mesh.h>
#include <renderengine/PrismSpatialLayout.h>
#include <renderengine/Texture.h>
#include <renderengine/private/Description.h>
#include <sync/sync.h>
#include <ui/ColorSpace.h>
#include <ui/DebugUtils.h>
#include <ui/GraphicBuffer.h>
#include <ui/Rect.h>
#include <ui/Region.h>
#include <utils/KeyedVector.h>
#include <utils/Trace.h>
#include "GLESRenderEngine.h"
#include "GLExtensions.h"
#include "GLFramebuffer.h"
#include "GLImage.h"
#include "GLShadowVertexGenerator.h"
#include "Program.h"
#include "ProgramCache.h"
#include "filters/BlurFilter.h"

bool checkGlError(const char* op, int lineNumber) {
    bool errorFound = false;
    GLint error = glGetError();
    while (error != GL_NO_ERROR) {
        errorFound = true;
        error = glGetError();
        ALOGV("after %s() (line # %d) glError (0x%x)\n", op, lineNumber, error);
    }
    return errorFound;
}

static constexpr bool outputDebugPPMs = false;

void writePPM(const char* basename, GLuint width, GLuint height) {
    ALOGV("writePPM #%s: %d x %d", basename, width, height);

    std::vector<GLubyte> pixels(width * height * 4);
    std::vector<GLubyte> outBuffer(width * height * 3);

    // TODO(courtneygo): We can now have float formats, need
    // to remove this code or update to support.
    // Make returned pixels fit in uint32_t, one byte per component
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    if (checkGlError(__FUNCTION__, __LINE__)) {
        return;
    }

    std::string filename(basename);
    filename.append(".ppm");
    std::ofstream file(filename.c_str(), std::ios::binary);
    if (!file.is_open()) {
        ALOGE("Unable to open file: %s", filename.c_str());
        ALOGE("You may need to do: \"adb shell setenforce 0\" to enable "
              "surfaceflinger to write debug images");
        return;
    }

    file << "P6\n";
    file << width << "\n";
    file << height << "\n";
    file << 255 << "\n";

    auto ptr = reinterpret_cast<char*>(pixels.data());
    auto outPtr = reinterpret_cast<char*>(outBuffer.data());
    for (int y = height - 1; y >= 0; y--) {
        char* data = ptr + y * width * sizeof(uint32_t);

        for (GLuint x = 0; x < width; x++) {
            // Only copy R, G and B components
            outPtr[0] = data[0];
            outPtr[1] = data[1];
            outPtr[2] = data[2];
            data += sizeof(uint32_t);
            outPtr += 3;
        }
    }
    file.write(reinterpret_cast<char*>(outBuffer.data()), outBuffer.size());
}

namespace android {
namespace renderengine {
namespace gl {

class BindNativeBufferAsFramebuffer {
public:
    BindNativeBufferAsFramebuffer(GLESRenderEngine& engine, ANativeWindowBuffer* buffer,
                                  const bool useFramebufferCache)
          : mEngine(engine), mFramebuffer(mEngine.getFramebufferForDrawing()), mStatus(NO_ERROR) {
        mStatus = mFramebuffer->setNativeWindowBuffer(buffer, mEngine.isProtected(),
                                                      useFramebufferCache)
                ? mEngine.bindFrameBuffer(mFramebuffer)
                : NO_MEMORY;
    }
    ~BindNativeBufferAsFramebuffer() {
        mFramebuffer->setNativeWindowBuffer(nullptr, false, /*arbitrary*/ true);
        mEngine.unbindFrameBuffer(mFramebuffer);
    }
    status_t getStatus() const { return mStatus; }

private:
    GLESRenderEngine& mEngine;
    Framebuffer* mFramebuffer;
    status_t mStatus;
};

using base::StringAppendF;
using ui::Dataspace;

namespace {

// PrismOS environment helper: float property with a default for empty/unparseable values.
float prismFloatProperty(const char* key, float defaultValue) {
    const std::string value = base::GetProperty(key, "");
    if (value.empty()) {
        return defaultValue;
    }
    const float parsed = strtof(value.c_str(), nullptr);
    return std::isfinite(parsed) ? parsed : defaultValue;
}

// PrismOS scene geometry, drawn through client-side arrays (the proven draw path on this
// driver — VBO-based draws of the environment silently do not render). The grid spans
// ±15m so it covers the whole eye FOV: at the 6m wall distance the visible half-width is
// ~7m, and a ±5m grid left the eye edges without reference lines.
constexpr size_t kPrismGridLines = 21;
constexpr size_t kPrismGridVertexCount = kPrismGridLines * 2 * 2;  // 2 directions, 2 points
constexpr size_t kPrismGridStrideFloats = 7;  // x, y, z, r, g, b, a
const std::array<GLfloat, kPrismGridVertexCount * kPrismGridStrideFloats> kPrismGridData = [] {
    std::array<GLfloat, kPrismGridVertexCount * kPrismGridStrideFloats> grid{};
    size_t i = 0;
    constexpr float kGridExtent = 15.0f;
    constexpr float kGridStep = 1.5f;
    constexpr float kGridDepth = -6.0f;
    for (size_t line = 0; line < kPrismGridLines; ++line) {
        const float fv = -kGridExtent + static_cast<float>(line) * kGridStep;
        grid[i++] = fv;         grid[i++] = -kGridExtent; grid[i++] = kGridDepth;
        grid[i++] = 0.18f; grid[i++] = 0.25f; grid[i++] = 0.35f; grid[i++] = 1.0f;
        grid[i++] = fv;         grid[i++] = kGridExtent;  grid[i++] = kGridDepth;
        grid[i++] = 0.18f; grid[i++] = 0.25f; grid[i++] = 0.35f; grid[i++] = 1.0f;
        grid[i++] = -kGridExtent; grid[i++] = fv;       grid[i++] = kGridDepth;
        grid[i++] = 0.18f; grid[i++] = 0.25f; grid[i++] = 0.35f; grid[i++] = 1.0f;
        grid[i++] = kGridExtent;  grid[i++] = fv;       grid[i++] = kGridDepth;
        grid[i++] = 0.18f; grid[i++] = 0.25f; grid[i++] = 0.35f; grid[i++] = 1.0f;
    }
    return grid;
}();
constexpr GLfloat kPrismSkyQuadPositions[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
// The unit world quad: x,y in [-0.5, 0.5], z = 0; uv with v=0 at the top.
constexpr GLfloat kPrismWorldQuadVertices[] = {
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
         0.5f, -0.5f, 0.0f, 1.0f, 1.0f,
        -0.5f,  0.5f, 0.0f, 0.0f, 0.0f,
         0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
};

GLuint compilePrismShader(GLenum type, const char* source) {
    const GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE) {
        char log[512] = {};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        ALOGE("PrismOS scene shader compilation failed: %s", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool prismLayerHasBuffer(const LayerSettings& layer) {
    return layer.source.buffer.buffer != nullptr &&
            layer.source.buffer.buffer->getBuffer() != nullptr;
}

// The spatial window only takes over layers that the regular path would blit verbatim; layers
// that drive the blur pipeline stay on the flat path. A tagged window whose 2D visible region
// is empty (another activity covered it) is still a candidate and keeps its last buffer in the
// world: background apps must remain spatially visible for multitasking, while the flat path
// continues to honor skipContentDraw for them.
bool prismLayerIsSpatialCandidate(const LayerSettings& layer) {
    return prismLayerHasBuffer(layer) && layer.backgroundBlurRadius <= 0 &&
            layer.blurRegions.empty();
}

// Layer names carry a "#<sequence>" suffix that changes between boots, so an exact match is
// accepted as well as the stable prefix in front of that suffix.
bool prismLayerNameMatches(const std::string& layerName, const std::string& requested) {
    if (requested.empty() || layerName.empty()) {
        return false;
    }
    if (layerName == requested) {
        return true;
    }
    return layerName.size() > requested.size() &&
            layerName.compare(0, requested.size(), requested) == 0 &&
            layerName[requested.size()] == '#';
}

const PrismSpatialWindowSettings* findPrismSpatialWindowSettings(
        const PrismSceneSettings& scene, const std::string& layerName) {
    for (const auto& settings : scene.spatialWindows) {
        if (prismLayerNameMatches(layerName, settings.id)) {
            return &settings;
        }
    }
    return nullptr;
}

// Prism spatialization is intentionally opt-in. WindowManager marks only buffer-bearing main
// application surfaces, while the legacy name selector bridges frames before metadata arrives.
// There is no largest-buffer fallback: a container, shadow, or system surface must never become a
// spatial window merely because it has the biggest buffer.
bool isPrismSpatialLayerSelected(const LayerSettings& layer, const PrismSceneSettings& scene,
                                 bool hasTaggedLayers) {
    if (!prismLayerIsSpatialCandidate(layer)) {
        return false;
    }
    if (layer.prismSpatialWindow) {
        return true;
    }
    return !hasTaggedLayers && !scene.spatialWindowLayer.empty() &&
            prismLayerNameMatches(layer.name, scene.spatialWindowLayer);
}

} // namespace

static status_t selectConfigForAttribute(EGLDisplay dpy, EGLint const* attrs, EGLint attribute,
                                         EGLint wanted, EGLConfig* outConfig) {
    EGLint numConfigs = -1, n = 0;
    eglGetConfigs(dpy, nullptr, 0, &numConfigs);
    std::vector<EGLConfig> configs(numConfigs, EGL_NO_CONFIG_KHR);
    eglChooseConfig(dpy, attrs, configs.data(), configs.size(), &n);
    configs.resize(n);

    if (!configs.empty()) {
        if (attribute != EGL_NONE) {
            for (EGLConfig config : configs) {
                EGLint value = 0;
                eglGetConfigAttrib(dpy, config, attribute, &value);
                if (wanted == value) {
                    *outConfig = config;
                    return NO_ERROR;
                }
            }
        } else {
            // just pick the first one
            *outConfig = configs[0];
            return NO_ERROR;
        }
    }

    return NAME_NOT_FOUND;
}

static status_t selectEGLConfig(EGLDisplay display, EGLint format, EGLint renderableType,
                                EGLConfig* config) {
    // select our EGLConfig. It must support EGL_RECORDABLE_ANDROID if
    // it is to be used with WIFI displays
    status_t err;
    EGLint wantedAttribute;
    EGLint wantedAttributeValue;

    std::vector<EGLint> attribs;
    if (renderableType) {
        const ui::PixelFormat pixelFormat = static_cast<ui::PixelFormat>(format);
        const bool is1010102 = pixelFormat == ui::PixelFormat::RGBA_1010102;

        // Default to 8 bits per channel.
        const EGLint tmpAttribs[] = {
                EGL_RENDERABLE_TYPE,
                renderableType,
                EGL_RECORDABLE_ANDROID,
                EGL_TRUE,
                EGL_SURFACE_TYPE,
                EGL_WINDOW_BIT | EGL_PBUFFER_BIT,
                EGL_FRAMEBUFFER_TARGET_ANDROID,
                EGL_TRUE,
                EGL_RED_SIZE,
                is1010102 ? 10 : 8,
                EGL_GREEN_SIZE,
                is1010102 ? 10 : 8,
                EGL_BLUE_SIZE,
                is1010102 ? 10 : 8,
                EGL_ALPHA_SIZE,
                is1010102 ? 2 : 8,
                EGL_NONE,
        };
        std::copy(tmpAttribs, tmpAttribs + (sizeof(tmpAttribs) / sizeof(EGLint)),
                  std::back_inserter(attribs));
        wantedAttribute = EGL_NONE;
        wantedAttributeValue = EGL_NONE;
    } else {
        // if no renderable type specified, fallback to a simplified query
        wantedAttribute = EGL_NATIVE_VISUAL_ID;
        wantedAttributeValue = format;
    }

    err = selectConfigForAttribute(display, attribs.data(), wantedAttribute, wantedAttributeValue,
                                   config);
    if (err == NO_ERROR) {
        EGLint caveat;
        if (eglGetConfigAttrib(display, *config, EGL_CONFIG_CAVEAT, &caveat))
            ALOGW_IF(caveat == EGL_SLOW_CONFIG, "EGL_SLOW_CONFIG selected!");
    }

    return err;
}

std::optional<RenderEngine::ContextPriority> GLESRenderEngine::createContextPriority(
        const RenderEngineCreationArgs& args) {
    if (!GLExtensions::getInstance().hasContextPriority()) {
        return std::nullopt;
    }

    switch (args.contextPriority) {
        case RenderEngine::ContextPriority::REALTIME:
            if (gl::GLExtensions::getInstance().hasRealtimePriority()) {
                return RenderEngine::ContextPriority::REALTIME;
            } else {
                ALOGI("Realtime priority unsupported, degrading gracefully to high priority");
                return RenderEngine::ContextPriority::HIGH;
            }
        case RenderEngine::ContextPriority::HIGH:
        case RenderEngine::ContextPriority::MEDIUM:
        case RenderEngine::ContextPriority::LOW:
            return args.contextPriority;
        default:
            return std::nullopt;
    }
}

std::unique_ptr<GLESRenderEngine> GLESRenderEngine::create(const RenderEngineCreationArgs& args) {
    // initialize EGL for the default display
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(display, nullptr, nullptr)) {
        LOG_ALWAYS_FATAL("failed to initialize EGL. EGL error=0x%x", eglGetError());
    }

    const auto eglVersion = eglQueryString(display, EGL_VERSION);
    if (!eglVersion) {
        checkGlError(__FUNCTION__, __LINE__);
        LOG_ALWAYS_FATAL("eglQueryString(EGL_VERSION) failed");
    }

    // Use the Android impl to grab EGL_NV_context_priority_realtime
    const auto eglExtensions = eglQueryString(display, EGL_EXTENSIONS);
    if (!eglExtensions) {
        checkGlError(__FUNCTION__, __LINE__);
        LOG_ALWAYS_FATAL("eglQueryString(EGL_EXTENSIONS) failed");
    }

    GLExtensions& extensions = GLExtensions::getInstance();
    extensions.initWithEGLStrings(eglVersion, eglExtensions);

    // The code assumes that ES2 or later is available if this extension is
    // supported.
    EGLConfig config = EGL_NO_CONFIG;
    if (!extensions.hasNoConfigContext()) {
        config = chooseEglConfig(display, args.pixelFormat, /*logConfig*/ true);
    }

    const std::optional<RenderEngine::ContextPriority> priority = createContextPriority(args);
    EGLContext protectedContext = EGL_NO_CONTEXT;
    if (args.enableProtectedContext && extensions.hasProtectedContent()) {
        protectedContext =
                createEglContext(display, config, nullptr, priority, Protection::PROTECTED);
        ALOGE_IF(protectedContext == EGL_NO_CONTEXT, "Can't create protected context");
    }

    EGLContext ctxt =
            createEglContext(display, config, protectedContext, priority, Protection::UNPROTECTED);

    // if can't create a GL context, we can only abort.
    LOG_ALWAYS_FATAL_IF(ctxt == EGL_NO_CONTEXT, "EGLContext creation failed");

    EGLSurface stub = EGL_NO_SURFACE;
    if (!extensions.hasSurfacelessContext()) {
        stub = createStubEglPbufferSurface(display, config, args.pixelFormat,
                                           Protection::UNPROTECTED);
        LOG_ALWAYS_FATAL_IF(stub == EGL_NO_SURFACE, "can't create stub pbuffer");
    }
    EGLBoolean success = eglMakeCurrent(display, stub, stub, ctxt);
    LOG_ALWAYS_FATAL_IF(!success, "can't make stub pbuffer current");
    extensions.initWithGLStrings(glGetString(GL_VENDOR), glGetString(GL_RENDERER),
                                 glGetString(GL_VERSION), glGetString(GL_EXTENSIONS));

    EGLSurface protectedStub = EGL_NO_SURFACE;
    if (protectedContext != EGL_NO_CONTEXT && !extensions.hasSurfacelessContext()) {
        protectedStub = createStubEglPbufferSurface(display, config, args.pixelFormat,
                                                    Protection::PROTECTED);
        ALOGE_IF(protectedStub == EGL_NO_SURFACE, "can't create protected stub pbuffer");
    }

    // now figure out what version of GL did we actually get
    GlesVersion version = parseGlesVersion(extensions.getVersion());

    LOG_ALWAYS_FATAL_IF(args.supportsBackgroundBlur && version < GLES_VERSION_3_0,
        "Blurs require OpenGL ES 3.0. Please unset ro.surface_flinger.supports_background_blur");

    // initialize the renderer while GL is current
    std::unique_ptr<GLESRenderEngine> engine;
    switch (version) {
        case GLES_VERSION_1_0:
        case GLES_VERSION_1_1:
            LOG_ALWAYS_FATAL("SurfaceFlinger requires OpenGL ES 2.0 minimum to run.");
            break;
        case GLES_VERSION_2_0:
        case GLES_VERSION_3_0:
            engine = std::make_unique<GLESRenderEngine>(args, display, config, ctxt, stub,
                                                        protectedContext, protectedStub);
            break;
    }

    ALOGI("OpenGL ES informations:");
    ALOGI("vendor    : %s", extensions.getVendor());
    ALOGI("renderer  : %s", extensions.getRenderer());
    ALOGI("version   : %s", extensions.getVersion());
    ALOGI("extensions: %s", extensions.getExtensions());
    ALOGI("GL_MAX_TEXTURE_SIZE = %zu", engine->getMaxTextureSize());
    ALOGI("GL_MAX_VIEWPORT_DIMS = %zu", engine->getMaxViewportDims());
    return engine;
}

EGLConfig GLESRenderEngine::chooseEglConfig(EGLDisplay display, int format, bool logConfig) {
    status_t err;
    EGLConfig config;

    // First try to get an ES3 config
    err = selectEGLConfig(display, format, EGL_OPENGL_ES3_BIT, &config);
    if (err != NO_ERROR) {
        // If ES3 fails, try to get an ES2 config
        err = selectEGLConfig(display, format, EGL_OPENGL_ES2_BIT, &config);
        if (err != NO_ERROR) {
            // If ES2 still doesn't work, probably because we're on the emulator.
            // try a simplified query
            ALOGW("no suitable EGLConfig found, trying a simpler query");
            err = selectEGLConfig(display, format, 0, &config);
            if (err != NO_ERROR) {
                // this EGL is too lame for android
                LOG_ALWAYS_FATAL("no suitable EGLConfig found, giving up");
            }
        }
    }

    if (logConfig) {
        // print some debugging info
        EGLint r, g, b, a;
        eglGetConfigAttrib(display, config, EGL_RED_SIZE, &r);
        eglGetConfigAttrib(display, config, EGL_GREEN_SIZE, &g);
        eglGetConfigAttrib(display, config, EGL_BLUE_SIZE, &b);
        eglGetConfigAttrib(display, config, EGL_ALPHA_SIZE, &a);
        ALOGI("EGL information:");
        ALOGI("vendor    : %s", eglQueryString(display, EGL_VENDOR));
        ALOGI("version   : %s", eglQueryString(display, EGL_VERSION));
        ALOGI("extensions: %s", eglQueryString(display, EGL_EXTENSIONS));
        ALOGI("Client API: %s", eglQueryString(display, EGL_CLIENT_APIS) ?: "Not Supported");
        ALOGI("EGLSurface: %d-%d-%d-%d, config=%p", r, g, b, a, config);
    }

    return config;
}

GLESRenderEngine::GLESRenderEngine(const RenderEngineCreationArgs& args, EGLDisplay display,
                                   EGLConfig config, EGLContext ctxt, EGLSurface stub,
                                   EGLContext protectedContext, EGLSurface protectedStub)
      : RenderEngine(args.renderEngineType),
        mEGLDisplay(display),
        mEGLConfig(config),
        mEGLContext(ctxt),
        mStubSurface(stub),
        mProtectedEGLContext(protectedContext),
        mProtectedStubSurface(protectedStub),
        mVpWidth(0),
        mVpHeight(0),
        mFramebufferImageCacheSize(args.imageCacheSize),
        mUseColorManagement(args.useColorManagement),
        mPrecacheToneMapperShaderOnly(args.precacheToneMapperShaderOnly) {
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &mMaxTextureSize);
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, mMaxViewportDims);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);

    // Initialize protected EGL Context.
    if (mProtectedEGLContext != EGL_NO_CONTEXT) {
        EGLBoolean success = eglMakeCurrent(display, mProtectedStubSurface, mProtectedStubSurface,
                                            mProtectedEGLContext);
        ALOGE_IF(!success, "can't make protected context current");
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        success = eglMakeCurrent(display, mStubSurface, mStubSurface, mEGLContext);
        LOG_ALWAYS_FATAL_IF(!success, "can't make default context current");
    }

    // mColorBlindnessCorrection = M;

    if (mUseColorManagement) {
        const ColorSpace srgb(ColorSpace::sRGB());
        const ColorSpace displayP3(ColorSpace::DisplayP3());
        const ColorSpace bt2020(ColorSpace::BT2020());

        // no chromatic adaptation needed since all color spaces use D65 for their white points.
        mSrgbToXyz = mat4(srgb.getRGBtoXYZ());
        mDisplayP3ToXyz = mat4(displayP3.getRGBtoXYZ());
        mBt2020ToXyz = mat4(bt2020.getRGBtoXYZ());
        mXyzToSrgb = mat4(srgb.getXYZtoRGB());
        mXyzToDisplayP3 = mat4(displayP3.getXYZtoRGB());
        mXyzToBt2020 = mat4(bt2020.getXYZtoRGB());

        // Compute sRGB to Display P3 and BT2020 transform matrix.
        // NOTE: For now, we are limiting output wide color space support to
        // Display-P3 and BT2020 only.
        mSrgbToDisplayP3 = mXyzToDisplayP3 * mSrgbToXyz;
        mSrgbToBt2020 = mXyzToBt2020 * mSrgbToXyz;

        // Compute Display P3 to sRGB and BT2020 transform matrix.
        mDisplayP3ToSrgb = mXyzToSrgb * mDisplayP3ToXyz;
        mDisplayP3ToBt2020 = mXyzToBt2020 * mDisplayP3ToXyz;

        // Compute BT2020 to sRGB and Display P3 transform matrix
        mBt2020ToSrgb = mXyzToSrgb * mBt2020ToXyz;
        mBt2020ToDisplayP3 = mXyzToDisplayP3 * mBt2020ToXyz;
    }

    char value[PROPERTY_VALUE_MAX];
    property_get("debug.egl.traceGpuCompletion", value, "0");
    if (atoi(value)) {
        mTraceGpuCompletion = true;
        mFlushTracer = std::make_unique<FlushTracer>(this);
    }

    if (args.supportsBackgroundBlur) {
        mBlurFilter = new BlurFilter(*this);
        checkErrors("BlurFilter creation");
    }

    mImageManager = std::make_unique<ImageManager>(this);
    mImageManager->initThread();
    mDrawingBuffer = createFramebuffer();
    sp<GraphicBuffer> buf =
            new GraphicBuffer(1, 1, PIXEL_FORMAT_RGBA_8888, 1,
                              GRALLOC_USAGE_HW_RENDER | GRALLOC_USAGE_HW_TEXTURE, "placeholder");

    const status_t err = buf->initCheck();
    if (err != OK) {
        ALOGE("Error allocating placeholder buffer: %d", err);
        return;
    }
    mPlaceholderBuffer = buf.get();
    EGLint attributes[] = {
            EGL_NONE,
    };
    mPlaceholderImage = eglCreateImageKHR(mEGLDisplay, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                                          mPlaceholderBuffer, attributes);
    ALOGE_IF(mPlaceholderImage == EGL_NO_IMAGE_KHR, "Failed to create placeholder image: %#x",
             eglGetError());

    mShadowTexture = std::make_unique<GLShadowTexture>();
}

GLESRenderEngine::~GLESRenderEngine() {
    // Destroy the image manager first.
    mImageManager = nullptr;
    mShadowTexture = nullptr;
    cleanFramebufferCache();
    ProgramCache::getInstance().purgeCaches();
    std::lock_guard<std::mutex> lock(mRenderingMutex);
    glDisableVertexAttribArray(Program::position);
    destroyPrismScene();
    unbindFrameBuffer(mDrawingBuffer.get());
    mDrawingBuffer = nullptr;
    eglDestroyImageKHR(mEGLDisplay, mPlaceholderImage);
    mImageCache.clear();
    if (mStubSurface != EGL_NO_SURFACE) {
        eglDestroySurface(mEGLDisplay, mStubSurface);
    }
    if (mProtectedStubSurface != EGL_NO_SURFACE) {
        eglDestroySurface(mEGLDisplay, mProtectedStubSurface);
    }
    if (mEGLContext != EGL_NO_CONTEXT) {
        eglDestroyContext(mEGLDisplay, mEGLContext);
    }
    if (mProtectedEGLContext != EGL_NO_CONTEXT) {
        eglDestroyContext(mEGLDisplay, mProtectedEGLContext);
    }
    eglMakeCurrent(mEGLDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglTerminate(mEGLDisplay);
    eglReleaseThread();
}

std::unique_ptr<Framebuffer> GLESRenderEngine::createFramebuffer() {
    return std::make_unique<GLFramebuffer>(*this);
}

std::unique_ptr<Image> GLESRenderEngine::createImage() {
    return std::make_unique<GLImage>(*this);
}

Framebuffer* GLESRenderEngine::getFramebufferForDrawing() {
    return mDrawingBuffer.get();
}

std::future<void> GLESRenderEngine::primeCache() {
    ProgramCache::getInstance().primeCache(mInProtectedContext ? mProtectedEGLContext : mEGLContext,
                                           mUseColorManagement, mPrecacheToneMapperShaderOnly);
    return {};
}

base::unique_fd GLESRenderEngine::flush() {
    ATRACE_CALL();
    if (!GLExtensions::getInstance().hasNativeFenceSync()) {
        return base::unique_fd();
    }

    EGLSyncKHR sync = eglCreateSyncKHR(mEGLDisplay, EGL_SYNC_NATIVE_FENCE_ANDROID, nullptr);
    if (sync == EGL_NO_SYNC_KHR) {
        ALOGW("failed to create EGL native fence sync: %#x", eglGetError());
        return base::unique_fd();
    }

    // native fence fd will not be populated until flush() is done.
    glFlush();

    // get the fence fd
    base::unique_fd fenceFd(eglDupNativeFenceFDANDROID(mEGLDisplay, sync));
    eglDestroySyncKHR(mEGLDisplay, sync);
    if (fenceFd == EGL_NO_NATIVE_FENCE_FD_ANDROID) {
        ALOGW("failed to dup EGL native fence sync: %#x", eglGetError());
    }

    // Only trace if we have a valid fence, as current usage falls back to
    // calling finish() if the fence fd is invalid.
    if (CC_UNLIKELY(mTraceGpuCompletion && mFlushTracer) && fenceFd.get() >= 0) {
        mFlushTracer->queueSync(eglCreateSyncKHR(mEGLDisplay, EGL_SYNC_FENCE_KHR, nullptr));
    }

    return fenceFd;
}

bool GLESRenderEngine::finish() {
    ATRACE_CALL();
    if (!GLExtensions::getInstance().hasFenceSync()) {
        ALOGW("no synchronization support");
        return false;
    }

    EGLSyncKHR sync = eglCreateSyncKHR(mEGLDisplay, EGL_SYNC_FENCE_KHR, nullptr);
    if (sync == EGL_NO_SYNC_KHR) {
        ALOGW("failed to create EGL fence sync: %#x", eglGetError());
        return false;
    }

    if (CC_UNLIKELY(mTraceGpuCompletion && mFlushTracer)) {
        mFlushTracer->queueSync(eglCreateSyncKHR(mEGLDisplay, EGL_SYNC_FENCE_KHR, nullptr));
    }

    return waitSync(sync, EGL_SYNC_FLUSH_COMMANDS_BIT_KHR);
}

bool GLESRenderEngine::waitSync(EGLSyncKHR sync, EGLint flags) {
    EGLint result = eglClientWaitSyncKHR(mEGLDisplay, sync, flags, 2000000000 /*2 sec*/);
    EGLint error = eglGetError();
    eglDestroySyncKHR(mEGLDisplay, sync);
    if (result != EGL_CONDITION_SATISFIED_KHR) {
        if (result == EGL_TIMEOUT_EXPIRED_KHR) {
            ALOGW("fence wait timed out");
        } else {
            ALOGW("error waiting on EGL fence: %#x", error);
        }
        return false;
    }

    return true;
}

bool GLESRenderEngine::waitFence(base::unique_fd fenceFd) {
    if (!GLExtensions::getInstance().hasNativeFenceSync() ||
        !GLExtensions::getInstance().hasWaitSync()) {
        return false;
    }

    // release the fd and transfer the ownership to EGLSync
    EGLint attribs[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fenceFd.release(), EGL_NONE};
    EGLSyncKHR sync = eglCreateSyncKHR(mEGLDisplay, EGL_SYNC_NATIVE_FENCE_ANDROID, attribs);
    if (sync == EGL_NO_SYNC_KHR) {
        ALOGE("failed to create EGL native fence sync: %#x", eglGetError());
        return false;
    }

    // XXX: The spec draft is inconsistent as to whether this should return an
    // EGLint or void.  Ignore the return value for now, as it's not strictly
    // needed.
    eglWaitSyncKHR(mEGLDisplay, sync, 0);
    EGLint error = eglGetError();
    eglDestroySyncKHR(mEGLDisplay, sync);
    if (error != EGL_SUCCESS) {
        ALOGE("failed to wait for EGL native fence sync: %#x", error);
        return false;
    }

    return true;
}

void GLESRenderEngine::clearWithColor(float red, float green, float blue, float alpha) {
    ATRACE_CALL();
    glDisable(GL_BLEND);
    glClearColor(red, green, blue, alpha);
    glClear(GL_COLOR_BUFFER_BIT);
}

void GLESRenderEngine::fillRegionWithColor(const Region& region, float red, float green, float blue,
                                           float alpha) {
    size_t c;
    Rect const* r = region.getArray(&c);
    Mesh mesh = Mesh::Builder()
                        .setPrimitive(Mesh::TRIANGLES)
                        .setVertices(c * 6 /* count */, 2 /* size */)
                        .build();
    Mesh::VertexArray<vec2> position(mesh.getPositionArray<vec2>());
    for (size_t i = 0; i < c; i++, r++) {
        position[i * 6 + 0].x = r->left;
        position[i * 6 + 0].y = r->top;
        position[i * 6 + 1].x = r->left;
        position[i * 6 + 1].y = r->bottom;
        position[i * 6 + 2].x = r->right;
        position[i * 6 + 2].y = r->bottom;
        position[i * 6 + 3].x = r->left;
        position[i * 6 + 3].y = r->top;
        position[i * 6 + 4].x = r->right;
        position[i * 6 + 4].y = r->bottom;
        position[i * 6 + 5].x = r->right;
        position[i * 6 + 5].y = r->top;
    }
    setupFillWithColor(red, green, blue, alpha);
    drawMesh(mesh);
}

void GLESRenderEngine::setScissor(const Rect& region) {
    glScissor(region.left, region.top, region.getWidth(), region.getHeight());
    glEnable(GL_SCISSOR_TEST);
}

void GLESRenderEngine::disableScissor() {
    glDisable(GL_SCISSOR_TEST);
}

void GLESRenderEngine::genTextures(size_t count, uint32_t* names) {
    glGenTextures(count, names);
}

void GLESRenderEngine::deleteTextures(size_t count, uint32_t const* names) {
    for (int i = 0; i < count; ++i) {
        mTextureView.erase(names[i]);
    }
    glDeleteTextures(count, names);
}

void GLESRenderEngine::bindExternalTextureImage(uint32_t texName, const Image& image) {
    ATRACE_CALL();
    const GLImage& glImage = static_cast<const GLImage&>(image);
    const GLenum target = GL_TEXTURE_EXTERNAL_OES;

    glBindTexture(target, texName);
    if (glImage.getEGLImage() != EGL_NO_IMAGE_KHR) {
        glEGLImageTargetTexture2DOES(target, static_cast<GLeglImageOES>(glImage.getEGLImage()));
    }
}

void GLESRenderEngine::bindExternalTextureBuffer(uint32_t texName, const sp<GraphicBuffer>& buffer,
                                                 const sp<Fence>& bufferFence) {
    ATRACE_CALL();

    bool found = false;
    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        auto cachedImage = mImageCache.find(buffer->getId());
        found = (cachedImage != mImageCache.end());
    }

    // If we couldn't find the image in the cache at this time, then either
    // SurfaceFlinger messed up registering the buffer ahead of time or we got
    // backed up creating other EGLImages.
    if (!found) {
        status_t cacheResult = mImageManager->cache(buffer);
        if (cacheResult != NO_ERROR) {
            ALOGE("Error with caching buffer: %d", cacheResult);
            return;
        }
    }

    // Whether or not we needed to cache, re-check mImageCache to make sure that
    // there's an EGLImage. The current threading model guarantees that we don't
    // destroy a cached image until it's really not needed anymore (i.e. this
    // function should not be called), so the only possibility is that something
    // terrible went wrong and we should just bind something and move on.
    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        auto cachedImage = mImageCache.find(buffer->getId());

        if (cachedImage == mImageCache.end()) {
            // We failed creating the image if we got here, so bail out.
            ALOGE("Failed to create an EGLImage when rendering");
            bindExternalTextureImage(texName, *createImage());
            return;
        }

        bindExternalTextureImage(texName, *cachedImage->second);
        mTextureView.insert_or_assign(texName, buffer->getId());
    }

    // Wait for the new buffer to be ready.
    if (bufferFence != nullptr && bufferFence->isValid()) {
        if (GLExtensions::getInstance().hasWaitSync()) {
            base::unique_fd fenceFd(bufferFence->dup());
            if (fenceFd == -1) {
                ALOGE("error dup'ing fence fd: %d", errno);
                return;
            }
            if (!waitFence(std::move(fenceFd))) {
                ALOGE("failed to wait on fence fd");
                return;
            }
        } else {
            status_t err = bufferFence->waitForever("RenderEngine::bindExternalTextureBuffer");
            if (err != NO_ERROR) {
                ALOGE("error waiting for fence: %d", err);
                return;
            }
        }
    }

    return;
}

void GLESRenderEngine::mapExternalTextureBuffer(const sp<GraphicBuffer>& buffer,
                                                bool /*isRenderable*/) {
    ATRACE_CALL();
    mImageManager->cacheAsync(buffer, nullptr);
}

std::shared_ptr<ImageManager::Barrier> GLESRenderEngine::cacheExternalTextureBufferForTesting(
        const sp<GraphicBuffer>& buffer) {
    auto barrier = std::make_shared<ImageManager::Barrier>();
    mImageManager->cacheAsync(buffer, barrier);
    return barrier;
}

status_t GLESRenderEngine::cacheExternalTextureBufferInternal(const sp<GraphicBuffer>& buffer) {
    if (buffer == nullptr) {
        return BAD_VALUE;
    }

    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        if (mImageCache.count(buffer->getId()) > 0) {
            // If there's already an image then fail fast here.
            return NO_ERROR;
        }
    }
    ATRACE_CALL();

    // Create the image without holding a lock so that we don't block anything.
    std::unique_ptr<Image> newImage = createImage();

    bool created = newImage->setNativeWindowBuffer(buffer->getNativeBuffer(),
                                                   buffer->getUsage() & GRALLOC_USAGE_PROTECTED);
    if (!created) {
        ALOGE("Failed to create image. id=%" PRIx64 " size=%ux%u st=%u usage=%#" PRIx64 " fmt=%d",
              buffer->getId(), buffer->getWidth(), buffer->getHeight(), buffer->getStride(),
              buffer->getUsage(), buffer->getPixelFormat());
        return NO_INIT;
    }

    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        if (mImageCache.count(buffer->getId()) > 0) {
            // In theory it's possible for another thread to recache the image,
            // so bail out if another thread won.
            return NO_ERROR;
        }
        mImageCache.insert(std::make_pair(buffer->getId(), std::move(newImage)));
    }

    return NO_ERROR;
}

void GLESRenderEngine::unmapExternalTextureBuffer(const sp<GraphicBuffer>& buffer) {
    mImageManager->releaseAsync(buffer->getId(), nullptr);
}

std::shared_ptr<ImageManager::Barrier> GLESRenderEngine::unbindExternalTextureBufferForTesting(
        uint64_t bufferId) {
    auto barrier = std::make_shared<ImageManager::Barrier>();
    mImageManager->releaseAsync(bufferId, barrier);
    return barrier;
}

void GLESRenderEngine::unbindExternalTextureBufferInternal(uint64_t bufferId) {
    std::unique_ptr<Image> image;
    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        const auto& cachedImage = mImageCache.find(bufferId);

        if (cachedImage != mImageCache.end()) {
            ALOGV("Destroying image for buffer: %" PRIu64, bufferId);
            // Move the buffer out of cache first, so that we can destroy
            // without holding the cache's lock.
            image = std::move(cachedImage->second);
            mImageCache.erase(bufferId);
            return;
        }
    }
    ALOGV("Failed to find image for buffer: %" PRIu64, bufferId);
}

int GLESRenderEngine::getContextPriority() {
    int value;
    eglQueryContext(mEGLDisplay, mEGLContext, EGL_CONTEXT_PRIORITY_LEVEL_IMG, &value);
    return value;
}

FloatRect GLESRenderEngine::setupLayerCropping(const LayerSettings& layer, Mesh& mesh) {
    // Translate win by the rounded corners rect coordinates, to have all values in
    // layer coordinate space.
    FloatRect cropWin = layer.geometry.boundaries;
    const FloatRect& roundedCornersCrop = layer.geometry.roundedCornersCrop;
    cropWin.left -= roundedCornersCrop.left;
    cropWin.right -= roundedCornersCrop.left;
    cropWin.top -= roundedCornersCrop.top;
    cropWin.bottom -= roundedCornersCrop.top;
    Mesh::VertexArray<vec2> cropCoords(mesh.getCropCoordArray<vec2>());
    cropCoords[0] = vec2(cropWin.left, cropWin.top);
    cropCoords[1] = vec2(cropWin.left, cropWin.top + cropWin.getHeight());
    cropCoords[2] = vec2(cropWin.right, cropWin.top + cropWin.getHeight());
    cropCoords[3] = vec2(cropWin.right, cropWin.top);

    setupCornerRadiusCropSize(roundedCornersCrop.getWidth(), roundedCornersCrop.getHeight());
    return cropWin;
}

void GLESRenderEngine::handleRoundedCorners(const DisplaySettings& display,
                                            const LayerSettings& layer, const Mesh& mesh) {
    // We separate the layer into 3 parts essentially, such that we only turn on blending for the
    // top rectangle and the bottom rectangle, and turn off blending for the middle rectangle.
    FloatRect bounds = layer.geometry.roundedCornersCrop;

    // Explicitly compute the transform from the clip rectangle to the physical
    // display. Normally, this is done in glViewport but we explicitly compute
    // it here so that we can get the scissor bounds correct.
    const Rect& source = display.clip;
    const Rect& destination = display.physicalDisplay;
    // Here we compute the following transform:
    // 1. Translate the top left corner of the source clip to (0, 0)
    // 2. Rotate the clip rectangle about the origin in accordance with the
    // orientation flag
    // 3. Translate the top left corner back to the origin.
    // 4. Scale the clip rectangle to the destination rectangle dimensions
    // 5. Translate the top left corner to the destination rectangle's top left
    // corner.
    const mat4 translateSource = mat4::translate(vec4(-source.left, -source.top, 0, 1));
    mat4 rotation;
    int displacementX = 0;
    int displacementY = 0;
    float destinationWidth = static_cast<float>(destination.getWidth());
    float destinationHeight = static_cast<float>(destination.getHeight());
    float sourceWidth = static_cast<float>(source.getWidth());
    float sourceHeight = static_cast<float>(source.getHeight());
    const float rot90InRadians = 2.0f * static_cast<float>(M_PI) / 4.0f;
    switch (display.orientation) {
        case ui::Transform::ROT_90:
            rotation = mat4::rotate(rot90InRadians, vec3(0, 0, 1));
            displacementX = source.getHeight();
            std::swap(sourceHeight, sourceWidth);
            break;
        case ui::Transform::ROT_180:
            rotation = mat4::rotate(rot90InRadians * 2.0f, vec3(0, 0, 1));
            displacementY = source.getHeight();
            displacementX = source.getWidth();
            break;
        case ui::Transform::ROT_270:
            rotation = mat4::rotate(rot90InRadians * 3.0f, vec3(0, 0, 1));
            displacementY = source.getWidth();
            std::swap(sourceHeight, sourceWidth);
            break;
        default:
            break;
    }

    const mat4 intermediateTranslation = mat4::translate(vec4(displacementX, displacementY, 0, 1));
    const mat4 scale = mat4::scale(
            vec4(destinationWidth / sourceWidth, destinationHeight / sourceHeight, 1, 1));
    const mat4 translateDestination =
            mat4::translate(vec4(destination.left, destination.top, 0, 1));
    const mat4 globalTransform =
            translateDestination * scale * intermediateTranslation * rotation * translateSource;

    const mat4 transformMatrix = globalTransform * layer.geometry.positionTransform;
    const vec4 leftTopCoordinate(bounds.left, bounds.top, 1.0, 1.0);
    const vec4 rightBottomCoordinate(bounds.right, bounds.bottom, 1.0, 1.0);
    const vec4 leftTopCoordinateInBuffer = transformMatrix * leftTopCoordinate;
    const vec4 rightBottomCoordinateInBuffer = transformMatrix * rightBottomCoordinate;
    bounds = FloatRect(std::min(leftTopCoordinateInBuffer[0], rightBottomCoordinateInBuffer[0]),
                       std::min(leftTopCoordinateInBuffer[1], rightBottomCoordinateInBuffer[1]),
                       std::max(leftTopCoordinateInBuffer[0], rightBottomCoordinateInBuffer[0]),
                       std::max(leftTopCoordinateInBuffer[1], rightBottomCoordinateInBuffer[1]));

    // Finally, we cut the layer into 3 parts, with top and bottom parts having rounded corners
    // and the middle part without rounded corners.
    const int32_t radius = ceil(
            (layer.geometry.roundedCornersRadius.x + layer.geometry.roundedCornersRadius.y) / 2.0);
    const Rect topRect(bounds.left, bounds.top, bounds.right, bounds.top + radius);
    setScissor(topRect);
    drawMesh(mesh);
    const Rect bottomRect(bounds.left, bounds.bottom - radius, bounds.right, bounds.bottom);
    setScissor(bottomRect);
    drawMesh(mesh);

    // The middle part of the layer can turn off blending.
    if (topRect.bottom < bottomRect.top) {
        const Rect middleRect(bounds.left, bounds.top + radius, bounds.right,
                              bounds.bottom - radius);
        setScissor(middleRect);
        mState.cornerRadius = 0.0;
        disableBlending();
        drawMesh(mesh);
    }
    disableScissor();
}

status_t GLESRenderEngine::bindFrameBuffer(Framebuffer* framebuffer) {
    ATRACE_CALL();
    GLFramebuffer* glFramebuffer = static_cast<GLFramebuffer*>(framebuffer);
    EGLImageKHR eglImage = glFramebuffer->getEGLImage();
    uint32_t textureName = glFramebuffer->getTextureName();
    uint32_t framebufferName = glFramebuffer->getFramebufferName();

    // Bind the texture and turn our EGLImage into a texture
    glBindTexture(GL_TEXTURE_2D, textureName);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, (GLeglImageOES)eglImage);

    // Bind the Framebuffer to render into
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferName);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureName, 0);

    uint32_t glStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    ALOGE_IF(glStatus != GL_FRAMEBUFFER_COMPLETE_OES, "glCheckFramebufferStatusOES error %d",
             glStatus);

    return glStatus == GL_FRAMEBUFFER_COMPLETE_OES ? NO_ERROR : BAD_VALUE;
}

void GLESRenderEngine::unbindFrameBuffer(Framebuffer* /*framebuffer*/) {
    ATRACE_CALL();

    // back to main framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool GLESRenderEngine::canSkipPostRenderCleanup() const {
    return mPriorResourcesCleaned ||
            (mLastDrawFence != nullptr && mLastDrawFence->getStatus() != Fence::Status::Signaled);
}

void GLESRenderEngine::cleanupPostRender() {
    ATRACE_CALL();

    if (canSkipPostRenderCleanup()) {
        // If we don't have a prior frame needing cleanup, then don't do anything.
        return;
    }

    // Bind the texture to placeholder so that backing image data can be freed.
    GLFramebuffer* glFramebuffer = static_cast<GLFramebuffer*>(getFramebufferForDrawing());
    glFramebuffer->allocateBuffers(1, 1, mPlaceholderDrawBuffer);

    // Release the cached fence here, so that we don't churn reallocations when
    // we could no-op repeated calls of this method instead.
    mLastDrawFence = nullptr;
    mPriorResourcesCleaned = true;
}

void GLESRenderEngine::cleanFramebufferCache() {
    std::lock_guard<std::mutex> lock(mFramebufferImageCacheMutex);
    // Bind the texture to placeholder so that backing image data can be freed.
    GLFramebuffer* glFramebuffer = static_cast<GLFramebuffer*>(getFramebufferForDrawing());
    glFramebuffer->allocateBuffers(1, 1, mPlaceholderDrawBuffer);

    while (!mFramebufferImageCache.empty()) {
        EGLImageKHR expired = mFramebufferImageCache.front().second;
        mFramebufferImageCache.pop_front();
        eglDestroyImageKHR(mEGLDisplay, expired);
        DEBUG_EGL_IMAGE_TRACKER_DESTROY();
    }
}

void GLESRenderEngine::checkErrors() const {
    checkErrors(nullptr);
}

void GLESRenderEngine::checkErrors(const char* tag) const {
    do {
        // there could be more than one error flag
        GLenum error = glGetError();
        if (error == GL_NO_ERROR) break;
        if (tag == nullptr) {
            ALOGE("GL error 0x%04x", int(error));
        } else {
            ALOGE("GL error: %s -> 0x%04x", tag, int(error));
        }
    } while (true);
}

bool GLESRenderEngine::supportsProtectedContent() const {
    return mProtectedEGLContext != EGL_NO_CONTEXT;
}

void GLESRenderEngine::useProtectedContext(bool useProtectedContext) {
    if (useProtectedContext == mInProtectedContext ||
        (useProtectedContext && !supportsProtectedContent())) {
        return;
    }

    const EGLSurface surface = useProtectedContext ? mProtectedStubSurface : mStubSurface;
    const EGLContext context = useProtectedContext ? mProtectedEGLContext : mEGLContext;
    if (eglMakeCurrent(mEGLDisplay, surface, surface, context) == EGL_TRUE) {
        mInProtectedContext = useProtectedContext;
    }
}
EGLImageKHR GLESRenderEngine::createFramebufferImageIfNeeded(ANativeWindowBuffer* nativeBuffer,
                                                             bool isProtected,
                                                             bool useFramebufferCache) {
    sp<GraphicBuffer> graphicBuffer = GraphicBuffer::from(nativeBuffer);
    if (useFramebufferCache) {
        std::lock_guard<std::mutex> lock(mFramebufferImageCacheMutex);
        for (const auto& image : mFramebufferImageCache) {
            if (image.first == graphicBuffer->getId()) {
                return image.second;
            }
        }
    }
    EGLint attributes[] = {
            isProtected ? EGL_PROTECTED_CONTENT_EXT : EGL_NONE,
            isProtected ? EGL_TRUE : EGL_NONE,
            EGL_NONE,
    };
    EGLImageKHR image = eglCreateImageKHR(mEGLDisplay, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                                          nativeBuffer, attributes);
    if (useFramebufferCache) {
        if (image != EGL_NO_IMAGE_KHR) {
            std::lock_guard<std::mutex> lock(mFramebufferImageCacheMutex);
            if (mFramebufferImageCache.size() >= mFramebufferImageCacheSize) {
                EGLImageKHR expired = mFramebufferImageCache.front().second;
                mFramebufferImageCache.pop_front();
                eglDestroyImageKHR(mEGLDisplay, expired);
                DEBUG_EGL_IMAGE_TRACKER_DESTROY();
            }
            mFramebufferImageCache.push_back({graphicBuffer->getId(), image});
        }
    }

    if (image != EGL_NO_IMAGE_KHR) {
        DEBUG_EGL_IMAGE_TRACKER_CREATE();
    }
    return image;
}

uint32_t GLESRenderEngine::nextFreePrismLayoutSlot() const {
    std::unordered_set<uint32_t> usedSlots;
    usedSlots.reserve(mPrismSpatialWindowStates.size());
    for (const auto& [_, state] : mPrismSpatialWindowStates) {
        usedSlots.insert(state.automaticLayoutSlot);
    }
    uint32_t slot = 0;
    while (usedSlots.count(slot) > 0) {
        ++slot;
    }
    return slot;
}

std::vector<GLESRenderEngine::PrismSpatialLayer> GLESRenderEngine::resolvePrismSpatialLayers(
        const std::vector<LayerSettings>& layers, const PrismSceneSettings& scene) {
    std::vector<PrismSpatialLayer> resolved;
    if (!scene.spatialWindowEnabled) {
        return resolved;
    }

    const uint64_t frame = ++mPrismSceneFrame;
    const bool hasTaggedLayers = std::any_of(layers.begin(), layers.end(),
                                             [](const LayerSettings& layer) {
                                                 return layer.prismSpatialWindow;
                                             });
    // PrismOS bring-up diagnostics: once per second, report why tagged layers fail selection.
    if (frame % 60 == 1) {
        std::string diag = "layers=" + std::to_string(layers.size()) +
                " tagged=" + std::to_string(hasTaggedLayers);
        for (const auto& layer : layers) {
            if (!prismLayerHasBuffer(layer)) continue;
            diag += " | '" + layer.name + "' tag=" +
                    std::to_string(layer.prismSpatialWindow) + " skipDraw=" +
                    std::to_string(layer.skipContentDraw) + " blur=" +
                    std::to_string(layer.backgroundBlurRadius) + " id=" +
                    std::to_string(layer.prismSpatialWindowId);
        }
        ALOGD("PrismOS resolve: %s", diag.c_str());
    }
    for (const auto& layer : layers) {
        if (layer.prismSpatialWindow && !prismLayerIsSpatialCandidate(layer)) {
            ALOGV("PrismOS spatial window '%s' is temporarily unavailable: no usable buffer",
                  layer.name.c_str());
            auto state = mPrismSpatialWindowStates.find(
                    prism::spatialStateKey(layer.prismSpatialWindowId, layer.name));
            if (state != mPrismSpatialWindowStates.end()) {
                state->second.lastSeenFrame = frame;
            }
            continue;
        }
        if (!isPrismSpatialLayerSelected(layer, scene, hasTaggedLayers) || layer.name.empty()) {
            continue;
        }

        // The stereo packing emits two eye copies of every layer, sharing name, window id, and
        // buffer. Placement state persists across frames (an app switched away and back keeps
        // its slot), so frame-resolved dedup is keyed per frame: deduping on the map insert
        // alone would resolve a window only on the single frame its state was created.
        const std::string stateKey =
                prism::spatialStateKey(layer.prismSpatialWindowId, layer.name);
        auto [state, inserted] = mPrismSpatialWindowStates.try_emplace(stateKey);
        if (inserted) {
            const sp<GraphicBuffer> buffer = layer.source.buffer.buffer->getBuffer();
            const uint32_t slot = nextFreePrismLayoutSlot();
            state->second.settings =
                    prism::automaticLayout(slot, static_cast<float>(buffer->getWidth()),
                                           static_cast<float>(buffer->getHeight()), scene);
            state->second.settings.id = layer.name;
            state->second.automaticLayoutSlot = slot;
        }
        state->second.lastSeenFrame = frame;
        if (state->second.lastResolvedFrame == frame) {
            // The second SBS eye copy of the same window this frame: the scene draws each
            // window once per eye, so a second entry would double-blend it.
            continue;
        }
        state->second.lastResolvedFrame = frame;

        // Overrides are merged every frame so property-driven moves apply to windows that are
        // already seated; without one, the automatic layout is kept.
        PrismSpatialWindowSettings effective = state->second.settings;
        if (const auto* settingsOverride = findPrismSpatialWindowSettings(scene, layer.name)) {
            effective = *settingsOverride;
            effective.id = layer.name;
        }
        resolved.push_back({&layer, std::move(effective)});
    }

    for (auto state = mPrismSpatialWindowStates.begin();
         state != mPrismSpatialWindowStates.end();) {
        if (frame - state->second.lastSeenFrame > scene.spatialWindowPlacementRetentionFrames) {
            ALOGI("PrismOS spatial window placement '%s' expired; freeing its layout slot",
                  state->first.c_str());
            state = mPrismSpatialWindowStates.erase(state);
        } else {
            ++state;
        }
    }
    return resolved;
}

bool GLESRenderEngine::initializePrismScene() {
    if (mPrismSceneProgram != 0) {
        return true;
    }

    static constexpr const char* kVertexShader = R"(
        attribute vec3 position;
        attribute vec4 color;
        uniform mat4 mvp;
        varying lowp vec4 vColor;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            vColor = color;
        }
    )";
    static constexpr const char* kFragmentShader = R"(
        precision mediump float;
        varying lowp vec4 vColor;
        void main() {
            gl_FragColor = vColor;
        }
    )";

    const GLuint vertexShader = compilePrismShader(GL_VERTEX_SHADER, kVertexShader);
    const GLuint fragmentShader = compilePrismShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (vertexShader == 0 || fragmentShader == 0) {
        if (vertexShader != 0) glDeleteShader(vertexShader);
        if (fragmentShader != 0) glDeleteShader(fragmentShader);
        return false;
    }

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE) {
        char log[512] = {};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        ALOGE("PrismOS scene program link failed: %s", log);
        glDeleteProgram(program);
        return false;
    }

    mPrismSceneMvpLocation = glGetUniformLocation(program, "mvp");
    mPrismScenePositionLocation = glGetAttribLocation(program, "position");
    mPrismSceneColorLocation = glGetAttribLocation(program, "color");
    if (mPrismSceneMvpLocation < 0 || mPrismScenePositionLocation < 0 ||
        mPrismSceneColorLocation < 0) {
        ALOGE("PrismOS scene program is missing an attribute or uniform");
        glDeleteProgram(program);
        mPrismSceneMvpLocation = -1;
        mPrismScenePositionLocation = -1;
        mPrismSceneColorLocation = -1;
        return false;
    }

    // PrismOS: the reference grid — the only decorative geometry, drawn every frame with the
    // scene program. It doubles as a "scene reaches the panel" marker. The early bring-up
    // cube buffer was removed: leaving it bound to GL_ARRAY_BUFFER after this function made
    // every later client-pointer draw in the scene read vertices from it, which rendered the
    // whole world empty.
    std::vector<GLfloat> grid;
    grid.reserve(44 * 7);
    const auto addGridVertex = [](std::vector<GLfloat>& vertices, float x, float y, float z) {
        vertices.insert(vertices.end(),
                        {x, y, z, 0.18f, 0.25f, 0.35f, 1.0f});
    };
    for (int i = -5; i <= 5; ++i) {
        addGridVertex(grid, static_cast<float>(i), -5.0f, -6.0f);
        addGridVertex(grid, static_cast<float>(i), 5.0f, -6.0f);
        addGridVertex(grid, -5.0f, static_cast<float>(i), -6.0f);
        addGridVertex(grid, 5.0f, static_cast<float>(i), -6.0f);
    }
    glGenBuffers(1, &mPrismSceneGridBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, mPrismSceneGridBuffer);
    glBufferData(GL_ARRAY_BUFFER, grid.size() * sizeof(GLfloat), grid.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    mPrismSceneProgram = program;
    return true;
}

void GLESRenderEngine::destroyPrismScene() {
    if (mPrismSceneGridBuffer != 0) {
        glDeleteBuffers(1, &mPrismSceneGridBuffer);
        mPrismSceneGridBuffer = 0;
    }
    if (mPrismSceneCubeBuffer != 0) {
        glDeleteBuffers(1, &mPrismSceneCubeBuffer);
        mPrismSceneCubeBuffer = 0;
    }
    if (mPrismSceneProgram != 0) {
        glDeleteProgram(mPrismSceneProgram);
        mPrismSceneProgram = 0;
    }
    if (mPrismSkyTexture != 0) {
        glDeleteTextures(1, &mPrismSkyTexture);
        mPrismSkyTexture = 0;
    }
    mPrismSkyLoadedPath.clear();
    if (mPrismMenuTexture != 0) {
        glDeleteTextures(1, &mPrismMenuTexture);
        mPrismMenuTexture = 0;
    }
    mPrismMenuLoadedPath.clear();
    for (auto& [path, envTexture] : mPrismEnvTextures) {
        if (envTexture.texture != 0) {
            glDeleteTextures(1, &envTexture.texture);
            envTexture.texture = 0;
        }
    }
    mPrismEnvTextures.clear();
    if (mPrismEnvQuadBuffer != 0) {
        glDeleteBuffers(1, &mPrismEnvQuadBuffer);
        mPrismEnvQuadBuffer = 0;
    }
    if (mPrismEnvQuadProgram != 0) {
        glDeleteProgram(mPrismEnvQuadProgram);
        mPrismEnvQuadProgram = 0;
    }
    if (mPrismSkyProgram != 0) {
        glDeleteProgram(mPrismSkyProgram);
        mPrismSkyProgram = 0;
    }
    if (mPrismMenuProgram != 0) {
        glDeleteProgram(mPrismMenuProgram);
        mPrismMenuProgram = 0;
    }
    mPrismEnvLoadAttempted = false;
}

// Loads a PrismOS raw texture: b"PRW1" (RGB888) or b"PRW2" (RGBA8888), then u32 width,
// u32 height (little endian), rows stored top-first. The channel count is taken from the magic.
bool GLESRenderEngine::loadPrismRawTexture(const std::string& path, GLuint* texture,
                                           uint32_t* outWidth, uint32_t* outHeight) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        ALOGW("PrismOS env: texture not found (%s)", path.c_str());
        return false;
    }
    char header[12];
    const bool headerOk = fread(header, 1, sizeof(header), file) == sizeof(header);
    uint32_t channels = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (headerOk) {
        if (memcmp(header, "PRW1", 4) == 0) {
            channels = 3;
        } else if (memcmp(header, "PRW2", 4) == 0) {
            channels = 4;
        }
        memcpy(&width, header + 4, sizeof(width));
        memcpy(&height, header + 8, sizeof(height));
    }
    if (!headerOk || channels == 0 || width == 0 || height == 0 || width > 8192 ||
        height > 8192) {
        ALOGW("PrismOS env: texture malformed (%s)", path.c_str());
        fclose(file);
        return false;
    }
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * channels);
    const bool readOk = fread(pixels.data(), 1, pixels.size(), file) == pixels.size();
    fclose(file);
    if (!readOk) {
        ALOGW("PrismOS env: texture truncated (%s)", path.c_str());
        return false;
    }

    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    const GLenum format = channels == 3 ? GL_RGB : GL_RGBA;
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE,
                 pixels.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    if (outWidth != nullptr) {
        *outWidth = width;
    }
    if (outHeight != nullptr) {
        *outHeight = height;
    }
    ALOGI("PrismOS env: texture %ux%u loaded from %s", width, height, path.c_str());
    return true;
}

// PrismOS test environment: compiles the sky, menu, and world-quad programs, and lazily loads
// the textures by path. Paths and the quad list come from the scene settings (SurfaceFlinger
// owns them: the packed scene file / vrender commands). Paths are re-checked whenever they
// change, so scenes can be swapped at runtime.
bool GLESRenderEngine::ensurePrismEnvironment(const PrismSceneSettings& scene) {
    if (!scene.envEnabled) {
        return false;
    }
    if (mPrismEnvQuadProgram == 0 || mPrismEnvQuadBuffer == 0) {
        if (mPrismEnvLoadAttempted) {
            return false;
        }
        mPrismEnvLoadAttempted = true;
    }

    // Fullscreen quad in NDC, shared by the sky and menu programs (positions only; the menu uv
    // is derived in its vertex shader from the NDC position).
    if (mPrismEnvQuadBuffer == 0) {
        static constexpr GLfloat kQuad[] = {
                -1.0f, -1.0f,
                 1.0f, -1.0f,
                -1.0f,  1.0f,
                 1.0f,  1.0f,
        };
        glGenBuffers(1, &mPrismEnvQuadBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, mPrismEnvQuadBuffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(kQuad), kQuad, GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // Sky: per-pixel ray from the eye rotation and the eye frustum, sampled in the
    // equirectangular panorama. World-locked, so it rotates with head tracking.
    // Drawn with CLIENT-SIDE vertex arrays: the VBO-based draws proved unreliable on this
    // driver (the client-pointer draws — the grid, the test quads — render fine), and the
    // scene geometry is a handful of quads where the client path costs nothing.
    static constexpr const char* kSkyVertexShader = R"(
        attribute vec2 position;
        uniform mat3 uSkyRotation;
        uniform vec2 uTanHalfFov;
        varying vec2 vSkyUv;
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            vec3 dirCam = normalize(vec3(position.x * uTanHalfFov.x,
                                         position.y * uTanHalfFov.y, -1.0));
            vec3 dir = uSkyRotation * dirCam;
            vSkyUv = vec2(atan(dir.z, -dir.x) * 0.15915494 + 0.5,
                          0.5 - asin(clamp(dir.y, -1.0, 1.0)) * 0.31830989);
        }
    )";
    static constexpr const char* kSkyFragmentShader = R"(
        precision mediump float;
        uniform sampler2D uSky;
        varying vec2 vSkyUv;
        void main() {
            gl_FragColor = texture2D(uSky, vSkyUv);
        }
    )";

    const GLuint skyVertex = compilePrismShader(GL_VERTEX_SHADER, kSkyVertexShader);
    const GLuint skyFragment = compilePrismShader(GL_FRAGMENT_SHADER, kSkyFragmentShader);
    if (skyVertex != 0 && skyFragment != 0) {
        const GLuint program = glCreateProgram();
        glAttachShader(program, skyVertex);
        glAttachShader(program, skyFragment);
        glLinkProgram(program);
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked == GL_TRUE) {
            mPrismSkyProgram = program;
            mPrismSkyPositionLocation = glGetAttribLocation(program, "position");
            mPrismSkyRotationLocation = glGetUniformLocation(program, "uSkyRotation");
            mPrismSkyTanHalfFovLocation = glGetUniformLocation(program, "uTanHalfFov");
            mPrismSkySamplerLocation = glGetUniformLocation(program, "uSky");
        } else {
            ALOGE("PrismOS env: sky program link failed");
            glDeleteProgram(program);
        }
    } else {
        if (skyVertex != 0) glDeleteShader(skyVertex);
        if (skyFragment != 0) glDeleteShader(skyFragment);
    }

    // Sky texture: loaded once per path; a vrender/scene path change reloads it.
    if (!scene.skyPath.empty() && mPrismSkyLoadedPath != scene.skyPath &&
        mPrismSkyAttemptedPath != scene.skyPath) {
        mPrismSkyAttemptedPath = scene.skyPath;
        GLuint texture = 0;
        if (loadPrismRawTexture(scene.skyPath, &texture, nullptr, nullptr)) {
            if (mPrismSkyTexture != 0) {
                glDeleteTextures(1, &mPrismSkyTexture);
            }
            mPrismSkyTexture = texture;
            mPrismSkyLoadedPath = scene.skyPath;
            ALOGI("PrismOS env: sky switched to %s", scene.skyPath.c_str());
        }
    }

    // Menu strip: RGBA overlay anchored to the top of each eye viewport. The uv is derived
    // from the NDC position, so the shared position-only quad works for it too.
    static constexpr const char* kMenuVertexShader = R"(
        attribute vec2 position;
        uniform vec2 uMenuScale;
        uniform vec2 uMenuOffset;
        varying vec2 vMenuUv;
        void main() {
            gl_Position = vec4(position.x, position.y * uMenuScale.y + uMenuOffset.y, 0.0, 1.0);
            vMenuUv = vec2(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
        }
    )";
    static constexpr const char* kMenuFragmentShader = R"(
        precision mediump float;
        uniform sampler2D uMenu;
        varying vec2 vMenuUv;
        void main() {
            gl_FragColor = texture2D(uMenu, vMenuUv);
        }
    )";

    const GLuint menuVertex = compilePrismShader(GL_VERTEX_SHADER, kMenuVertexShader);
    const GLuint menuFragment = compilePrismShader(GL_FRAGMENT_SHADER, kMenuFragmentShader);
    if (menuVertex != 0 && menuFragment != 0) {
        const GLuint program = glCreateProgram();
        glAttachShader(program, menuVertex);
        glAttachShader(program, menuFragment);
        glLinkProgram(program);
        GLint linked = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (linked == GL_TRUE) {
            mPrismMenuProgram = program;
            mPrismMenuPositionLocation = glGetAttribLocation(program, "position");
            mPrismMenuTextureLocation = glGetUniformLocation(program, "uMenu");
            mPrismMenuScaleLocation = glGetUniformLocation(program, "uMenuScale");
            mPrismMenuOffsetLocation = glGetUniformLocation(program, "uMenuOffset");
        } else {
            ALOGE("PrismOS env: menu program link failed");
            glDeleteProgram(program);
        }
    } else {
        if (menuVertex != 0) glDeleteShader(menuVertex);
        if (menuFragment != 0) glDeleteShader(menuFragment);
    }

    // Menu texture: loaded once per path, same reload rule as the sky.
    if (!scene.menuPath.empty() && mPrismMenuLoadedPath != scene.menuPath &&
        mPrismMenuAttemptedPath != scene.menuPath) {
        mPrismMenuAttemptedPath = scene.menuPath;
        GLuint texture = 0;
        uint32_t menuWidth = 0;
        uint32_t menuHeight = 0;
        if (loadPrismRawTexture(scene.menuPath, &texture, &menuWidth, &menuHeight)) {
            if (mPrismMenuTexture != 0) {
                glDeleteTextures(1, &mPrismMenuTexture);
            }
            mPrismMenuTexture = texture;
            mPrismMenuLoadedPath = scene.menuPath;
            mPrismMenuLoadedWidth = menuWidth;
            mPrismMenuLoadedHeight = menuHeight;
            ALOGI("PrismOS env: menu switched to %s", scene.menuPath.c_str());
        }
    }

    // World-quad program + unit quad (positions + uv), drawn once per scene quad with a
    // per-quad world matrix.
    if (mPrismEnvQuadProgram == 0) {
        static constexpr const char* kEnvQuadVertexShader = R"(
            attribute vec3 position;
            attribute vec2 uv;
            uniform mat4 uMvp;
            varying vec2 vQuadUv;
            void main() {
                gl_Position = uMvp * vec4(position, 1.0);
                vQuadUv = uv;
            }
        )";
        static constexpr const char* kEnvQuadFragmentShader = R"(
            precision mediump float;
            uniform sampler2D uQuadTexture;
            varying vec2 vQuadUv;
            void main() {
                gl_FragColor = texture2D(uQuadTexture, vQuadUv);
            }
        )";

        const GLuint quadVertex = compilePrismShader(GL_VERTEX_SHADER, kEnvQuadVertexShader);
        const GLuint quadFragment =
                compilePrismShader(GL_FRAGMENT_SHADER, kEnvQuadFragmentShader);
        if (quadVertex != 0 && quadFragment != 0) {
            const GLuint program = glCreateProgram();
            glAttachShader(program, quadVertex);
            glAttachShader(program, quadFragment);
            glLinkProgram(program);
            GLint linked = GL_FALSE;
            glGetProgramiv(program, GL_LINK_STATUS, &linked);
            if (linked == GL_TRUE) {
                mPrismEnvQuadProgram = program;
                mPrismEnvQuadPositionLocation = glGetAttribLocation(program, "position");
                mPrismEnvQuadUvLocation = glGetAttribLocation(program, "uv");
                mPrismEnvQuadMvpLocation = glGetUniformLocation(program, "uMvp");
                mPrismEnvQuadTextureLocation = glGetUniformLocation(program, "uQuadTexture");
            } else {
                ALOGE("PrismOS env: quad program link failed");
                glDeleteProgram(program);
            }
        } else {
            if (quadVertex != 0) glDeleteShader(quadVertex);
            if (quadFragment != 0) glDeleteShader(quadFragment);
        }

        if (mPrismWorldQuadBuffer == 0) {
            // Unit quad in the quad's local plane, triangle strip; uv with v=0 at the top.
            static constexpr GLfloat kWorldQuad[] = {
                    -0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
                     0.5f, -0.5f, 0.0f, 1.0f, 1.0f,
                    -0.5f,  0.5f, 0.0f, 0.0f, 0.0f,
                     0.5f,  0.5f, 0.0f, 1.0f, 0.0f,
            };
            glGenBuffers(1, &mPrismWorldQuadBuffer);
            glBindBuffer(GL_ARRAY_BUFFER, mPrismWorldQuadBuffer);
            glBufferData(GL_ARRAY_BUFFER, sizeof(kWorldQuad), kWorldQuad, GL_STATIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }

    const bool skyReady = mPrismSkyProgram != 0 && mPrismSkyTexture != 0;
    ALOGI("PrismOS env: %s (sky=%d menu=%d)", skyReady ? "ready" : "unavailable",
          mPrismSkyTexture != 0 ? 1 : 0, mPrismMenuTexture != 0 ? 1 : 0);
    return skyReady;
}

// Draws the world-locked sky for the current eye viewport. Depth is left untouched so the
// window pass above it behaves exactly like the plain-clear case.
void GLESRenderEngine::drawPrismSkyBackdrop(const mat4& headView, float tanHalfFovY,
                                            float aspect) {
    if (!base::GetBoolProperty("debug.prismos.env.background", true)) {
        return;
    }
    if (mPrismSkyProgram == 0 || mPrismSkyTexture == 0) {
        return;
    }

    // View rotation (3-DOF) with an optional panorama yaw alignment. headView is column
    // major; its upper-left 3x3 is the world->camera rotation. The shader maps a camera-space
    // ray into the world, which needs the INVERSE (= transpose, rotations are orthogonal):
    // world = yaw * Rview^T * dirCam. Using Rview directly mirrors the panorama and makes it
    // spin counter to the head.
    const float* view = headView.asArray();
    const float yawDegrees =
            prismFloatProperty("debug.prismos.env.sky_yaw", 0.0f);
    const float yawRadians = yawDegrees * static_cast<float>(M_PI) / 180.0f;
    const float c = std::cos(yawRadians);
    const float s = std::sin(yawRadians);
    // Rview^T in column-major: column j of the transpose is row j of the original.
    const float t0 = view[0], t1 = view[4], t2 = view[8];
    const float t3 = view[1], t4 = view[5], t5 = view[9];
    const float t6 = view[2], t7 = view[6], t8 = view[10];
    GLfloat rotation[9];
    // M = yaw * Rview^T: column j of M is the yaw rotation applied to column j of Rview^T.
    rotation[0] = c * t0 + s * t2;
    rotation[1] = t1;
    rotation[2] = -s * t0 + c * t2;
    rotation[3] = c * t3 + s * t5;
    rotation[4] = t4;
    rotation[5] = -s * t3 + c * t5;
    rotation[6] = c * t6 + s * t8;
    rotation[7] = t7;
    rotation[8] = -s * t6 + c * t8;

    glUseProgram(mPrismSkyProgram);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mPrismSkyTexture);
    glUniform1i(mPrismSkySamplerLocation, 0);
    glUniformMatrix3fv(mPrismSkyRotationLocation, 1, GL_FALSE, rotation);
    glUniform2f(mPrismSkyTanHalfFovLocation, tanHalfFovY * aspect, tanHalfFovY);
    glBindBuffer(GL_ARRAY_BUFFER, mPrismEnvQuadBuffer);
    glEnableVertexAttribArray(static_cast<GLuint>(mPrismSkyPositionLocation));
    glVertexAttribPointer(static_cast<GLuint>(mPrismSkyPositionLocation), 2, GL_FLOAT,
                          GL_FALSE, 8, nullptr);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(static_cast<GLuint>(mPrismSkyPositionLocation));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}

// Draws the world-space textured quads of the loaded scene. Depth-tested and depth-written,
// so the quads sort against the spatial windows naturally; blending honors RGBA textures.
void GLESRenderEngine::drawPrismEnvQuads(const PrismSceneSettings& scene, const mat4& eyeMvp) {
    if (mPrismEnvQuadProgram == 0 || mPrismWorldQuadBuffer == 0) {
        return;
    }

    glUseProgram(mPrismEnvQuadProgram);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glActiveTexture(GL_TEXTURE0);
    // The unit quad comes from client-side memory (the proven draw path on this driver).
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableVertexAttribArray(static_cast<GLuint>(mPrismEnvQuadPositionLocation));
    glVertexAttribPointer(static_cast<GLuint>(mPrismEnvQuadPositionLocation), 3, GL_FLOAT,
                          GL_FALSE, 20, kPrismWorldQuadVertices);
    glEnableVertexAttribArray(static_cast<GLuint>(mPrismEnvQuadUvLocation));
    glVertexAttribPointer(static_cast<GLuint>(mPrismEnvQuadUvLocation), 2, GL_FLOAT, GL_FALSE,
                          20, reinterpret_cast<const void*>(12));
    glUniform1i(mPrismEnvQuadTextureLocation, 0);

    for (const auto& quad : scene.envQuads) {
        if (!quad.visible || quad.texturePath.empty() || quad.height <= 0.0f) {
            continue;
        }
        auto& envTexture = mPrismEnvTextures[quad.texturePath];
        if (envTexture.texture == 0) {
            GLuint texture = 0;
            uint32_t width = 0;
            uint32_t height = 0;
            if (!loadPrismRawTexture(quad.texturePath, &texture, &width, &height)) {
                mPrismEnvTextures.erase(quad.texturePath);
                continue;
            }
            envTexture.texture = texture;
            envTexture.width = width;
            envTexture.height = height;
        }
        glBindTexture(GL_TEXTURE_2D, envTexture.texture);
        glUniform1i(mPrismEnvQuadTextureLocation, 0);

        const float aspect = envTexture.height > 0
                ? static_cast<float>(envTexture.width) / static_cast<float>(envTexture.height)
                : 1.0f;
        const mat4 world =
                mat4::translate(vec4(quad.centerX, quad.centerY, -quad.distance, 1.0f)) *
                mat4::rotate(quad.yawDegrees * static_cast<float>(M_PI) / 180.0f,
                             vec3(0.0f, 1.0f, 0.0f)) *
                mat4::scale(vec4(quad.height * aspect, quad.height, 1.0f, 1.0f));
        glUniformMatrix4fv(mPrismEnvQuadMvpLocation, 1, GL_FALSE,
                           (eyeMvp * world).asArray());
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisable(GL_BLEND);
}

// Draws the RGBA menu strip (SteamVR-style bar) anchored to the top of the eye viewport.
void GLESRenderEngine::drawPrismMenuOverlay(float eyeViewportHeight) {
    if (mPrismMenuTexture == 0 || mPrismMenuProgram == 0) {
        return;
    }
    if (!base::GetBoolProperty("debug.prismos.env.menu_overlay", true)) {
        return;
    }

    // The bar height in NDC comes from the loaded strip's own aspect: the strip is authored at
    // mPrismMenuLoadedWidth x mPrismMenuLoadedHeight for an eye viewport of that same width,
    // anchored so its top sits at NDC +1.
    if (mPrismMenuLoadedWidth == 0 || mPrismMenuLoadedHeight == 0) {
        return;
    }
    const float barNdcHeight = 2.0f * static_cast<float>(mPrismMenuLoadedHeight) /
            static_cast<float>(mPrismMenuLoadedWidth);
    const float scaleY = barNdcHeight;
    const float offsetY = 1.0f - barNdcHeight;

    glUseProgram(mPrismMenuProgram);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, mPrismMenuTexture);
    glUniform1i(mPrismMenuTextureLocation, 0);
    glUniform2f(mPrismMenuScaleLocation, 1.0f, scaleY);
    glUniform2f(mPrismMenuOffsetLocation, 0.0f, offsetY);
    glEnableVertexAttribArray(static_cast<GLuint>(mPrismMenuPositionLocation));
    glVertexAttribPointer(static_cast<GLuint>(mPrismMenuPositionLocation), 2, GL_FLOAT,
                          GL_FALSE, 8, kPrismSkyQuadPositions);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisableVertexAttribArray(static_cast<GLuint>(mPrismMenuPositionLocation));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);
}

void GLESRenderEngine::getPrismGazeHit(bool& valid, std::string& windowName, float& u,
                                       float& v) {
    std::lock_guard lock(mPrismGazeHit.mutex);
    valid = mPrismGazeHit.valid;
    windowName = mPrismGazeHit.windowName;
    u = mPrismGazeHit.u;
    v = mPrismGazeHit.v;
}

bool GLESRenderEngine::drawPrismScene(
        const DisplaySettings& display, const PrismSceneSettings& scene,
        const std::vector<PrismSpatialLayer>& spatialLayers,
        std::vector<const LayerSettings*>* spatialWindowsDrawn) {
    if (spatialWindowsDrawn != nullptr) {
        spatialWindowsDrawn->clear();
    }
    if (!initializePrismScene()) {
        return false;
    }

    const GLint targetWidth = display.physicalDisplay.getWidth();
    const GLint targetHeight = display.physicalDisplay.getHeight();
    const GLint leftHeight = targetHeight / 2;
    const GLint rightHeight = targetHeight - leftHeight;
    if (targetWidth <= 0 || targetHeight <= 1 || leftHeight <= 0 || rightHeight <= 0) {
        return false;
    }

    static bool logged = false;
    if (!logged) {
        ALOGI("PRISM3D: scene enabled target %dx%d left viewport %dx%d right viewport %dx%d IPD %f",
              targetWidth, targetHeight, targetWidth, leftHeight, targetWidth, rightHeight,
              scene.ipd);
        logged = true;
    }

    GLint oldViewport[4] = {};
    GLint oldScissor[4] = {};
    GLint oldProgram = 0;
    GLint oldArrayBuffer = 0;
    GLint oldDepthFunc = GL_LESS;
    GLfloat oldLineWidth = 1.0f;
    GLfloat oldClearColor[4] = {};
    GLboolean oldDepthTest = GL_FALSE;
    GLboolean oldBlend = GL_FALSE;
    GLboolean oldScissorTest = GL_FALSE;
    GLboolean oldDepthMask = GL_TRUE;
    glGetIntegerv(GL_VIEWPORT, oldViewport);
    glGetIntegerv(GL_SCISSOR_BOX, oldScissor);
    glGetIntegerv(GL_CURRENT_PROGRAM, &oldProgram);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &oldArrayBuffer);
    glGetIntegerv(GL_DEPTH_FUNC, &oldDepthFunc);
    glGetFloatv(GL_LINE_WIDTH, &oldLineWidth);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, oldClearColor);
    glGetBooleanv(GL_DEPTH_WRITEMASK, &oldDepthMask);
    oldDepthTest = glIsEnabled(GL_DEPTH_TEST);
    oldBlend = glIsEnabled(GL_BLEND);
    oldScissorTest = glIsEnabled(GL_SCISSOR_TEST);

    const float fovRadians = scene.fovDegrees * 3.14159265358979323846f / 180.0f;
    const float focalLength = 1.0f / std::tan(fovRadians * 0.5f);
    const bool monoView = scene.monoView;
    // PrismOS: the projection always uses the single-eye aspect (one eye viewport is
    // targetWidth x leftHeight). The mono capture view reuses it so that the captured image
    // shows the same frustum the wearer sees; the eye image is stretched over the full
    // buffer height instead of shrinking the field of view and losing side windows.
    const float aspect = static_cast<float>(targetWidth) / static_cast<float>(leftHeight);
    const float zScale = (scene.farPlane + scene.nearPlane) / (scene.nearPlane - scene.farPlane);
    const float zTranslate = (2.0f * scene.farPlane * scene.nearPlane) /
            (scene.nearPlane - scene.farPlane);
    const mat4 projection(vec4(focalLength / aspect, 0.0f, 0.0f, 0.0f),
                          vec4(0.0f, focalLength, 0.0f, 0.0f),
                          vec4(0.0f, 0.0f, zScale, -1.0f),
                          vec4(0.0f, 0.0f, zTranslate, 0.0f));

    // headTransform is head space to world space, so invert it exactly once for the view.
    const mat4 headView = inverse(scene.headTransform);
    const auto drawSpatialWindow = [&](const PrismSpatialLayer& spatial, const mat4& eyeMvp) {
        const LayerSettings* layer = spatial.layer;
        if (layer == nullptr || !prismLayerIsSpatialCandidate(*layer)) {
            return false;
        }
        const sp<GraphicBuffer> windowBuffer = layer->source.buffer.buffer->getBuffer();
        const uint32_t bufferWidth = windowBuffer->getWidth();
        const uint32_t bufferHeight = windowBuffer->getHeight();
        if (bufferWidth == 0 || bufferHeight == 0) {
            return false;
        }

        validateInputBufferUsage(windowBuffer);
        bindExternalTextureBuffer(layer->source.buffer.textureName, windowBuffer,
                                  layer->source.buffer.fence);
        Texture windowTexture(Texture::TEXTURE_EXTERNAL, layer->source.buffer.textureName);
        windowTexture.setMatrix(layer->source.buffer.textureTransform.asArray());
        windowTexture.setFiltering(layer->source.buffer.useTextureFiltering);
        windowTexture.setDimensions(bufferWidth, bufferHeight);

        Mesh windowMesh = Mesh::Builder()
                                  .setPrimitive(Mesh::TRIANGLE_FAN)
                                  .setVertices(4 /* count */, 2 /* size */)
                                  .setTexCoords(2 /* size */)
                                  .build();
        Mesh::VertexArray<vec2> position(windowMesh.getPositionArray<vec2>());
        position[0] = vec2(0.0f, 0.0f);
        position[1] = vec2(0.0f, static_cast<float>(bufferHeight));
        position[2] = vec2(static_cast<float>(bufferWidth), static_cast<float>(bufferHeight));
        position[3] = vec2(static_cast<float>(bufferWidth), 0.0f);
        Mesh::VertexArray<vec2> texCoords(windowMesh.getTexCoordArray<vec2>());
        texCoords[0] = vec2(0.0f, 0.0f);
        texCoords[1] = vec2(0.0f, 1.0f);
        texCoords[2] = vec2(1.0f, 1.0f);
        texCoords[3] = vec2(1.0f, 0.0f);

        // PrismOS bring-up: the window bound to the camera. The window content is drawn
        // fullscreen in NDC (the world transform bypassed) — isolates the texture path from
        // the world transform math.
        if (base::GetBoolProperty("debug.prismos.env.window_camera_test", false)) {
            ALOGD("PrismOS camtest: drawing '%s' fullscreen NDC, buf=%ux%u",
                  layer->name.c_str(), bufferWidth, bufferHeight);
            position[0] = vec2(-1.0f, 1.0f);
            position[1] = vec2(-1.0f, -1.0f);
            position[2] = vec2(1.0f, -1.0f);
            position[3] = vec2(1.0f, 1.0f);
            texCoords[0] = vec2(0.0f, 0.0f);
            texCoords[1] = vec2(0.0f, 1.0f);
            texCoords[2] = vec2(1.0f, 1.0f);
            texCoords[3] = vec2(1.0f, 0.0f);
            mState.projectionMatrix = mat4();
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            drawMesh(windowMesh);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);
            return true;
        }

        // PrismOS: the world frame (with the headset mount baked into the view rotation)
        // mirrors screen-space X relative to the window's local frame, so the sampled
        // content appears flipped left-to-right. Compensate in the UVs.
        texCoords[0] = vec2(1.0f, 0.0f);
        texCoords[1] = vec2(1.0f, 1.0f);
        texCoords[2] = vec2(0.0f, 1.0f);
        texCoords[3] = vec2(0.0f, 0.0f);

        const float panelHeight = std::max(0.05f, spatial.settings.heightMeters);
        const float scale = panelHeight / static_cast<float>(bufferHeight);
        const float distance = std::max(0.25f, spatial.settings.distanceMeters);
        const mat4 worldFromLocal =
                mat4::translate(vec4(spatial.settings.centerXMeters, spatial.settings.centerYMeters,
                                     -distance, 1.0f)) *
                mat4::scale(vec4(scale, -scale, scale, 1.0f)) *
                mat4::translate(vec4(-static_cast<float>(bufferWidth) * 0.5f,
                                     -static_cast<float>(bufferHeight) * 0.5f, 0.0f, 1.0f));

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        mState.projectionMatrix = eyeMvp * worldFromLocal;
        mState.cornerRadius = 0.0f;
        mState.cropSize = half2(0.0f, 0.0f);
        mState.drawShadows = false;
        mState.colorMatrix = layer->colorTransform;
        mState.displayColorMatrix = mat4();
        mState.inputTransformMatrix = mat4();
        mState.outputTransformMatrix = mat4();
        mState.displayMaxLuminance = 0.0f;
        mState.maxMasteringLuminance = 0.0f;
        mState.maxContentLuminance = 0.0f;

        setupLayerTexturing(windowTexture);
        setupLayerBlending(layer->source.buffer.usePremultipliedAlpha,
                           layer->source.buffer.isOpaque, false /* disableTexture */,
                           half4(1.0f, 1.0f, 1.0f, layer->alpha), 0.0f);
        if (layer->disableBlending) {
            glDisable(GL_BLEND);
        }
        setSourceY410BT2020(layer->source.buffer.isY410BT2020);
        setSourceDataSpace(layer->sourceDataspace);
        // LEQUAL keeps coplanar quads composited by draw order instead of rejecting the second
        // one: an activity window and its dialog (or any window override sharing a slot) sit on
        // the same plane, and the later, higher-stacked draw must blend over the earlier one.
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        drawMesh(windowMesh);
        // PrismOS bring-up diagnostics: the window draw state, once per second.
        {
            static int windowDiagCount = 0;
            if (windowDiagCount++ % 60 == 0) {
                const GLenum drawError = glGetError();
                const mat4& mvp = mState.projectionMatrix;
                ALOGD("PrismOS window: '%s' err=0x%x buf=%ux%u alpha=%.2f opaque=%d "
                      "mvpT=(%.2f, %.2f, %.2f)",
                      layer->name.c_str(), drawError, bufferWidth, bufferHeight,
                      static_cast<float>(layer->alpha), layer->source.buffer.isOpaque ? 1 : 0,
                      mvp[0][3], mvp[1][3], mvp[2][3]);
            }
        }

        setSourceY410BT2020(false);
        setSourceDataSpace(Dataspace::UNKNOWN);
        disableTexturing();
        disableBlending();
        mState.isPremultipliedAlpha = false;
        mState.isOpaque = true;
        mState.color = half4(0.0f, 0.0f, 0.0f, 0.0f);
        mState.colorMatrix = mat4();
        mState.displayColorMatrix = mat4();
        mState.projectionMatrix = eyeMvp;
        return true;
    };

    // PrismOS immersive fullscreen: draws the application buffer over the whole eye view
    // (both eyes get the same image on the stereo panel) with the shell hidden. The flat
    // 2D orientation is kept: no world transform, no mount mirror compensation.
    const auto drawSpatialWindowFullscreen = [&](const PrismSpatialLayer& spatial) {
        const LayerSettings* layer = spatial.layer;
        if (layer == nullptr || layer->source.buffer.buffer == nullptr) {
            return false;
        }
        const sp<GraphicBuffer> windowBuffer = layer->source.buffer.buffer->getBuffer();
        const uint32_t bufferWidth = windowBuffer->getWidth();
        const uint32_t bufferHeight = windowBuffer->getHeight();
        if (bufferWidth == 0 || bufferHeight == 0) {
            return false;
        }
        validateInputBufferUsage(windowBuffer);
        bindExternalTextureBuffer(layer->source.buffer.textureName, windowBuffer,
                                  layer->source.buffer.fence);
        Texture windowTexture(Texture::TEXTURE_EXTERNAL, layer->source.buffer.textureName);
        windowTexture.setMatrix(layer->source.buffer.textureTransform.asArray());
        windowTexture.setFiltering(layer->source.buffer.useTextureFiltering);
        windowTexture.setDimensions(bufferWidth, bufferHeight);

        Mesh windowMesh = Mesh::Builder()
                                  .setPrimitive(Mesh::TRIANGLE_FAN)
                                  .setVertices(4 /* count */, 2 /* size */)
                                  .setTexCoords(2 /* size */)
                                  .build();
        Mesh::VertexArray<vec2> position(windowMesh.getPositionArray<vec2>());
        position[0] = vec2(-1.0f, 1.0f);
        position[1] = vec2(-1.0f, -1.0f);
        position[2] = vec2(1.0f, -1.0f);
        position[3] = vec2(1.0f, 1.0f);
        Mesh::VertexArray<vec2> texCoords(windowMesh.getTexCoordArray<vec2>());
        texCoords[0] = vec2(0.0f, 0.0f);
        texCoords[1] = vec2(0.0f, 1.0f);
        texCoords[2] = vec2(1.0f, 1.0f);
        texCoords[3] = vec2(1.0f, 0.0f);

        mState.projectionMatrix = mat4();
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);
        setupLayerTexturing(windowTexture);
        setupLayerBlending(layer->source.buffer.usePremultipliedAlpha,
                           layer->source.buffer.isOpaque, false /* disableTexture */,
                           half4(1.0f, 1.0f, 1.0f, layer->alpha), 0.0f);
        drawMesh(windowMesh);
        disableTexturing();
        disableBlending();
        glDepthMask(GL_TRUE);
        glEnable(GL_DEPTH_TEST);
        return true;
    };

    const auto drawEye = [&](GLint viewportY, GLint viewportHeight, float eyeOffset) {
        glViewport(0, viewportY, targetWidth, viewportHeight);
        glScissor(0, viewportY, targetWidth, viewportHeight);
        glEnable(GL_SCISSOR_TEST);
        glClearColor(0.008f, 0.012f, 0.025f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // PrismOS: compile the environment programs and load the textures once, before the
        // sky/menu/quad draws that depend on them.
        ensurePrismEnvironment(scene);
        // PrismOS bring-up: unmistakable test pattern. Left eye = red, right eye = blue; two
        // quads drawn per eye: one with the scene program, one with a freshly compiled minimal
        // program. Logs the GL error and program validation to isolate invisible draws.
        if (base::GetBoolProperty("debug.prismos.env.testpattern", false)) {
            static const GLfloat kTestPositions[] = {
                    -0.5f, -0.5f, 0.0f,
                     0.5f, -0.5f, 0.0f,
                    -0.5f,  0.5f, 0.0f,
                     0.5f,  0.5f, 0.0f,
            };
            static const GLfloat kTestColors[] = {
                    0.0f, 1.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 1.0f,
                    0.0f, 1.0f, 0.0f, 1.0f,
            };
            const bool leftEye = viewportY == 0;
            glClearColor(leftEye ? 1.0f : 0.0f, 0.0f, leftEye ? 0.0f : 1.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);

            glUseProgram(mPrismSceneProgram);
            glUniformMatrix4fv(mPrismSceneMvpLocation, 1, GL_FALSE, mat4().asArray());
            glEnableVertexAttribArray(static_cast<GLuint>(mPrismScenePositionLocation));
            glEnableVertexAttribArray(static_cast<GLuint>(mPrismSceneColorLocation));
            glVertexAttribPointer(static_cast<GLuint>(mPrismScenePositionLocation), 3, GL_FLOAT,
                                  GL_FALSE, 0, kTestPositions);
            glVertexAttribPointer(static_cast<GLuint>(mPrismSceneColorLocation), 4, GL_FLOAT,
                                  GL_FALSE, 0, kTestColors);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            const GLenum sceneError = glGetError();

            // Fresh minimal program: same quad, offset to the left, cyan.
            static const GLuint freshProgram = [] {
                const char* vertex = "attribute vec2 p; void main() { gl_Position = "
                                     "vec4(p, 0.0, 1.0); }";
                const char* fragment = "precision mediump float; void main() { gl_FragColor = "
                                       "vec4(0.0, 1.0, 1.0, 1.0); }";
                const GLuint v = compilePrismShader(GL_VERTEX_SHADER, vertex);
                const GLuint f = compilePrismShader(GL_FRAGMENT_SHADER, fragment);
                const GLuint program = glCreateProgram();
                glAttachShader(program, v);
                glAttachShader(program, f);
                glLinkProgram(program);
                glDeleteShader(v);
                glDeleteShader(f);
                return program;
            }();
            glUseProgram(freshProgram);
            static const GLfloat kFreshPositions[] = {
                    -0.95f, -0.2f, 0.0f,
                    -0.45f, -0.2f, 0.0f,
                    -0.95f,  0.2f, 0.0f,
                    -0.45f,  0.2f, 0.0f,
            };
            const GLint freshLocation = glGetAttribLocation(freshProgram, "p");
            glEnableVertexAttribArray(static_cast<GLuint>(freshLocation));
            glVertexAttribPointer(static_cast<GLuint>(freshLocation), 2, GL_FLOAT, GL_FALSE, 0,
                                  kFreshPositions);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            const GLenum freshError = glGetError();
            glDisableVertexAttribArray(static_cast<GLuint>(freshLocation));

            GLint validateStatus = 0;
            glValidateProgram(mPrismSceneProgram);
            glGetProgramiv(mPrismSceneProgram, GL_VALIDATE_STATUS, &validateStatus);
            char validateLog[256] = {};
            glGetProgramInfoLog(mPrismSceneProgram, sizeof(validateLog), nullptr, validateLog);
            ALOGD("PrismOS test: eye=%s sceneErr=0x%x freshErr=0x%x sceneProg=%u valid=%d "
                  "log='%s'",
                  leftEye ? "L" : "R", sceneError, freshError, mPrismSceneProgram,
                  validateStatus, validateLog);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);
            return;
        }
        // PrismOS test environment: world-locked SteamVR-style sky, the reference grid, scene
        // quads, and the menu strip on top of everything (screen-space overlay).
        const mat4 view = mat4::translate(vec4(eyeOffset, 0.0f, 0.0f, 1.0f)) * headView;
        const mat4 eyeMvp = projection * view;

        // PrismOS immersive fullscreen: the immersive application fills the eye; sky,
        // grid, quads, other windows and the menu strip are skipped. Other tasks keep
        // their surfaces untouched and return to the scene when immersive is turned off.
        bool immersiveDrawn = false;
        if (!scene.immersiveWindow.empty()) {
            for (auto spatial = spatialLayers.rbegin(); spatial != spatialLayers.rend();
                 ++spatial) {
                if (prismLayerNameMatches(spatial->layer->name, scene.immersiveWindow)) {
                    immersiveDrawn = drawSpatialWindowFullscreen(*spatial);
                }
            }
        }
        if (immersiveDrawn) {
            return;
        }
        drawPrismSkyBackdrop(headView, 1.0f / focalLength,
                             static_cast<float>(targetWidth) / static_cast<float>(viewportHeight));
        // The reference grid: always drawn (it is the "scene reaches the panel" marker and a
        // depth cue); sits at z=-6, behind every window and quad.
        glUseProgram(mPrismSceneProgram);
        glUniformMatrix4fv(mPrismSceneMvpLocation, 1, GL_FALSE, eyeMvp.asArray());
        glEnableVertexAttribArray(static_cast<GLuint>(mPrismScenePositionLocation));
        glEnableVertexAttribArray(static_cast<GLuint>(mPrismSceneColorLocation));
        // Client-side arrays (the grid data is re-sent per draw; it is tiny and this path
        // proved reliable on this driver where VBO-based draws silently do not render).
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glVertexAttribPointer(static_cast<GLuint>(mPrismScenePositionLocation), 3, GL_FLOAT,
                              GL_FALSE, 28, kPrismGridData.data());
        glVertexAttribPointer(static_cast<GLuint>(mPrismSceneColorLocation), 4, GL_FLOAT,
                              GL_FALSE, 28, kPrismGridData.data() + 3);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(kPrismGridVertexCount));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        drawPrismEnvQuads(scene, eyeMvp);
        glUseProgram(mPrismSceneProgram);
        glEnableVertexAttribArray(static_cast<GLuint>(mPrismScenePositionLocation));
        glEnableVertexAttribArray(static_cast<GLuint>(mPrismSceneColorLocation));
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
        glDepthMask(GL_TRUE);
        glUniformMatrix4fv(mPrismSceneMvpLocation, 1, GL_FALSE, eyeMvp.asArray());
        // Painter order: windows resolved from the layer stack arrive top-first, so drawing
        // them back-to-front with LEQUAL gives stacked coplanar windows correct blending while
        // the depth test still sorts windows that sit at different world depths.
        for (auto spatial = spatialLayers.rbegin(); spatial != spatialLayers.rend(); ++spatial) {
            if (drawSpatialWindow(*spatial, eyeMvp) && spatialWindowsDrawn != nullptr) {
                spatialWindowsDrawn->push_back(spatial->layer);
            }
        }
        drawPrismMenuOverlay(static_cast<float>(viewportHeight));
    };

    if (monoView) {
        // Capture path: a single centered view over the whole buffer — what the wearer
        // sees, usable directly for screenshots and screen recording.
        drawEye(0, targetHeight, 0.0f);
    } else {
        drawEye(0, leftHeight, -scene.ipd * 0.5f);
        drawEye(leftHeight, rightHeight, scene.ipd * 0.5f);
    }

    // PrismOS spatial input: cast the gaze ray (head position + forward direction in the
    // world) against every spatial window plane and keep the nearest hit. The hit carries
    // the window name and the normalized in-window coordinates (u right, v down) — exactly
    // what the input side needs to convert into a touch on the application view.
    {
        const float* h = scene.headTransform.asArray();
        // headTransform is column major; direction = R * (0,0,-1), origin = R * (0,0,0,1).
        const float ox = h[12], oy = h[13], oz = h[14];
        float dx = -h[8], dy = -h[9], dz = -h[10];
        const float len = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (len > 1e-6f) {
            dx /= len;
            dy /= len;
            dz /= len;
        }
        bool hitValid = false;
        std::string hitName;
        float hitU = 0.5f, hitV = 0.5f, hitT = 1e9f;
        for (const auto& spatial : spatialLayers) {
            const auto& s = spatial.settings;
            if (dz > -1e-6f) {
                continue; // looking away from the window plane
            }
            const float t = (-s.distanceMeters - oz) / dz;
            if (t <= 0.0f || t >= hitT) {
                continue;
            }
            const float px = ox + dx * t - s.centerXMeters;
            const float py = oy + dy * t - s.centerYMeters;
            const float height = std::max(0.05f, s.heightMeters);
            const auto* layer = spatial.layer;
            const float width = layer != nullptr &&
                            layer->source.buffer.buffer != nullptr
                    ? height * static_cast<float>(layer->source.buffer.buffer->getBuffer()->getWidth()) /
                            static_cast<float>(layer->source.buffer.buffer->getBuffer()->getHeight())
                    : height;
            const float halfW = width * 0.5f, halfH = height * 0.5f;
            if (px < -halfW || px > halfW || py < -halfH || py > halfH) {
                continue;
            }
            hitValid = true;
            hitName = s.id;
            hitT = t;
            hitU = px / width + 0.5f;
            hitV = 0.5f - py / height;
        }
        {
            std::lock_guard lock(mPrismGazeHit.mutex);
            mPrismGazeHit.valid = hitValid;
            mPrismGazeHit.windowName = hitName;
            mPrismGazeHit.u = hitU;
            mPrismGazeHit.v = hitV;
        }
    }
    for (const auto& spatial : spatialLayers) {
        const sp<GraphicBuffer> buffer = spatial.layer->source.buffer.buffer->getBuffer();
        if (buffer->getUsage() & GRALLOC_USAGE_PROTECTED) {
            // Same rule as the flat path: do not keep protected EGLImages cached.
            unmapExternalTextureBuffer(buffer);
        }
    }

    glDisableVertexAttribArray(static_cast<GLuint>(mPrismScenePositionLocation));
    glDisableVertexAttribArray(static_cast<GLuint>(mPrismSceneColorLocation));
    glBindBuffer(GL_ARRAY_BUFFER, static_cast<GLuint>(oldArrayBuffer));
    glUseProgram(static_cast<GLuint>(oldProgram));
    glViewport(oldViewport[0], oldViewport[1], oldViewport[2], oldViewport[3]);
    glScissor(oldScissor[0], oldScissor[1], oldScissor[2], oldScissor[3]);
    if (oldScissorTest) {
        glEnable(GL_SCISSOR_TEST);
    } else {
        glDisable(GL_SCISSOR_TEST);
    }
    if (oldDepthTest) {
        glEnable(GL_DEPTH_TEST);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glDepthFunc(static_cast<GLenum>(oldDepthFunc));
    glDepthMask(oldDepthMask);
    glLineWidth(oldLineWidth);
    glClearColor(oldClearColor[0], oldClearColor[1], oldClearColor[2], oldClearColor[3]);
    if (oldBlend) {
        glEnable(GL_BLEND);
    } else {
        glDisable(GL_BLEND);
    }
    return true;
}

void GLESRenderEngine::drawLayersInternal(
        const std::shared_ptr<std::promise<RenderEngineResult>>&& resultPromise,
        const DisplaySettings& display, const std::vector<LayerSettings>& layers,
        const std::shared_ptr<ExternalTexture>& buffer, const bool useFramebufferCache,
        base::unique_fd&& bufferFence) {
    drawLayersInternalImpl(std::move(resultPromise), display, layers, buffer, useFramebufferCache,
                           std::move(bufferFence), nullptr);
}

void GLESRenderEngine::drawLayersInternalWithPrismScene(
        const std::shared_ptr<std::promise<RenderEngineResult>>&& resultPromise,
        const DisplaySettings& display, const std::vector<LayerSettings>& layers,
        const std::shared_ptr<ExternalTexture>& buffer, const bool useFramebufferCache,
        base::unique_fd&& bufferFence, const PrismSceneSettings& scene) {
    drawLayersInternalImpl(std::move(resultPromise), display, layers, buffer, useFramebufferCache,
                           std::move(bufferFence), &scene);
}

void GLESRenderEngine::drawLayersInternalImpl(
        const std::shared_ptr<std::promise<RenderEngineResult>>&& resultPromise,
        const DisplaySettings& display, const std::vector<LayerSettings>& layers,
        const std::shared_ptr<ExternalTexture>& buffer, const bool useFramebufferCache,
        base::unique_fd&& bufferFence, const PrismSceneSettings* scene) {
    ATRACE_CALL();
    // PrismOS: capture probe. Screenshot draws arrive with the fill layer only (≤1 layer);
    // rate-limited so ordinary frames stay silent. Remove once capture is fixed.
    if (layers.size() <= 1) {
        static std::chrono::steady_clock::time_point sLastProbe;
        const auto nowProbe = std::chrono::steady_clock::now();
        if (nowProbe - sLastProbe > std::chrono::seconds(1)) {
            sLastProbe = nowProbe;
            ALOGD("PrismOS drawLayers probe: layers=%zu display=[%d %d %d %d] scene=%s",
                  layers.size(), display.physicalDisplay.left, display.physicalDisplay.top,
                  display.physicalDisplay.right, display.physicalDisplay.bottom,
                  scene != nullptr ? "yes" : "no");
        }
    }
    const bool sceneEnabled = scene != nullptr && scene->enabled;
    const std::vector<PrismSpatialLayer> spatialLayers =
            sceneEnabled ? resolvePrismSpatialLayers(layers, *scene)
                         : std::vector<PrismSpatialLayer>();
    if (scene != nullptr && (!sceneEnabled || !scene->spatialWindowEnabled)) {
        // No GL resources or layer pointers are stored here, but clear placement state promptly
        // when the feature turns off. Capture-side draws pass scene == nullptr (screenshots,
        // the region sampler) and must not wipe placement state or membership logging.
        mPrismSpatialWindowStates.clear();
    }
    std::vector<const LayerSettings*> spatialWindowsDrawn;
    std::vector<std::string> spatialWindowsDrawnKeys;
    if (sceneEnabled && scene->spatialWindowEnabled) {
        std::string selected;
        for (const auto& spatial : spatialLayers) {
            if (!selected.empty()) selected += ", ";
            selected += spatial.settings.id;
        }
        if (selected != mPrismSpatialWindowLoggedLayers) {
            if (spatialLayers.empty()) {
                ALOGI("PrismOS spatial windows: none available this frame (%zu layers on screen)",
                      layers.size());
            } else {
                ALOGI("PrismOS spatial windows active: %s", selected.c_str());
            }
            mPrismSpatialWindowLoggedLayers = selected;
        }
    } else if (scene != nullptr && !mPrismSpatialWindowLoggedLayers.empty()) {
        ALOGI("PrismOS spatial windows disabled");
        mPrismSpatialWindowLoggedLayers.clear();
    }
    if (layers.empty() && !sceneEnabled) {
        ALOGV("Drawing empty layer stack");
        resultPromise->set_value({NO_ERROR, base::unique_fd()});
        return;
    }

    if (bufferFence.get() >= 0) {
        // Duplicate the fence for passing to waitFence.
        base::unique_fd bufferFenceDup(dup(bufferFence.get()));
        if (bufferFenceDup < 0 || !waitFence(std::move(bufferFenceDup))) {
            ATRACE_NAME("Waiting before draw");
            // PrismOS: a bounded wait. A dead producer (a force-stopped app mid-frame) would
            // otherwise hang the RenderEngine thread forever, freezing the whole display on
            // the last frame. A timed-out fence renders the buffer as-is for one frame.
            const int waitResult = sync_wait(bufferFence.get(), 250);
            if (waitResult != 0) {
                ALOGW("PrismOS: buffer fence wait timed out (%d), drawing anyway", waitResult);
            }
        }
    }

    if (buffer == nullptr) {
        ALOGE("No output buffer provided. Aborting GPU composition.");
        resultPromise->set_value({BAD_VALUE, base::unique_fd()});
        return;
    }

    validateOutputBufferUsage(buffer->getBuffer());

    std::unique_ptr<BindNativeBufferAsFramebuffer> fbo;
    // Gathering layers that requested blur, we'll need them to decide when to render to an
    // offscreen buffer, and when to render to the native buffer.
    std::deque<const LayerSettings> blurLayers;
    if (CC_LIKELY(mBlurFilter != nullptr)) {
        for (const auto& layer : layers) {
            if (layer.backgroundBlurRadius > 0) {
                blurLayers.push_back(layer);
            }
        }
    }
    const auto blurLayersSize = blurLayers.size();

    if (blurLayersSize == 0) {
        fbo = std::make_unique<BindNativeBufferAsFramebuffer>(*this,
                                                              buffer->getBuffer()
                                                                      .get()
                                                                      ->getNativeBuffer(),
                                                              useFramebufferCache);
        if (fbo->getStatus() != NO_ERROR) {
            ALOGE("Failed to bind framebuffer! Aborting GPU composition for buffer (%p).",
                  buffer->getBuffer()->handle);
            checkErrors();
            resultPromise->set_value({fbo->getStatus(), base::unique_fd()});
            return;
        }
        setViewportAndProjection(display.physicalDisplay, display.clip);
    } else {
        setViewportAndProjection(display.physicalDisplay, display.clip);
        auto status =
                mBlurFilter->setAsDrawTarget(display, blurLayers.front().backgroundBlurRadius);
        if (status != NO_ERROR) {
            ALOGE("Failed to prepare blur filter! Aborting GPU composition for buffer (%p).",
                  buffer->getBuffer()->handle);
            checkErrors();
            resultPromise->set_value({status, base::unique_fd()});
            return;
        }
    }

    // clear the entire buffer, sometimes when we reuse buffers we'd persist
    // ghost images otherwise.
    // we also require a full transparent framebuffer for overlays. This is
    // probably not quite efficient on all GPUs, since we could filter out
    // opaque layers.
    clearWithColor(0.0, 0.0, 0.0, 0.0);

    setOutputDataSpace(display.outputDataspace);
    setDisplayMaxLuminance(display.maxLuminance);
    setDisplayColorTransform(display.colorTransform);

    if (sceneEnabled) {
        auto* framebuffer = static_cast<GLFramebuffer*>(getFramebufferForDrawing());
        const uint32_t width = static_cast<uint32_t>(display.physicalDisplay.getWidth());
        const uint32_t height = static_cast<uint32_t>(display.physicalDisplay.getHeight());
        bool sceneRendered = false;
        if (width > 0 && height > 0 && framebuffer->prepareDepthBuffer(width, height)) {
            sceneRendered = drawPrismScene(display, *scene, spatialLayers, &spatialWindowsDrawn);
            framebuffer->detachDepthBuffer();
        }
        if (!sceneRendered) {
            spatialWindowsDrawn.clear();
            ALOGW("PrismOS 3D scene failed; continuing with Android layer composition");
        }
        // The stereo packing duplicates every layer into an eye copy sharing the window's name
        // and id, so flat-copy suppression is keyed by window identity: pointer identity would
        // leak the right-eye copy of every spatial window over the scene.
        spatialWindowsDrawnKeys.reserve(spatialWindowsDrawn.size());
        for (const LayerSettings* drawn : spatialWindowsDrawn) {
            spatialWindowsDrawnKeys.push_back(
                    prism::spatialStateKey(drawn->prismSpatialWindowId, drawn->name));
        }
        // The scene pass changes the GL viewport, ProgramCache state, and display transform.
        // Restore the ordinary Android path before entering the existing draw loop.
        setViewportAndProjection(display.physicalDisplay, display.clip);
        setDisplayColorTransform(display.colorTransform);
        setDisplayMaxLuminance(display.maxLuminance);
    }

    const mat4 projectionMatrix =
            ui::Transform(display.orientation).asMatrix4() * mState.projectionMatrix;

    Mesh mesh = Mesh::Builder()
                        .setPrimitive(Mesh::TRIANGLE_FAN)
                        .setVertices(4 /* count */, 2 /* size */)
                        .setTexCoords(2 /* size */)
                        .setCropCoords(2 /* size */)
                        .build();
    // Suppress only a layer whose window identity was actually drawn into the scene and that
    // opted into suppression. The key match covers both SBS eye copies of the same window while
    // leaving a different window with a similar name on the flat path.
    const auto shouldSuppressFlatCopy = [&](const LayerSettings& layer) {
        if (spatialWindowsDrawnKeys.empty()) {
            return false;
        }
        const std::string layerKey =
                prism::spatialStateKey(layer.prismSpatialWindowId, layer.name);
        if (std::find(spatialWindowsDrawnKeys.begin(), spatialWindowsDrawnKeys.end(), layerKey) ==
            spatialWindowsDrawnKeys.end()) {
            return false;
        }
        const auto spatial = std::find_if(spatialLayers.begin(), spatialLayers.end(),
                                          [&layerKey](const PrismSpatialLayer& candidate) {
                                              return prism::spatialStateKey(
                                                             candidate.layer->prismSpatialWindowId,
                                                             candidate.layer->name) == layerKey;
                                          });
        return spatial != spatialLayers.end() && spatial->settings.suppressFlatCopy;
    };
    const bool prismFlatLayersEnabled =
            base::GetBoolProperty("debug.prismos.flat_layers", true);
    for (const auto& layer : layers) {
        // PrismOS scene-only mode: the spatial scene (sky, grid, quads, windows) replaces the
        // flat Android UI entirely. The flat draws of heavy UI content through the stereo
        // packing were hanging the GPU driver, so they are skipped wholesale when disabled.
        if (sceneEnabled && !prismFlatLayersEnabled) {
            continue;
        }
        if (shouldSuppressFlatCopy(layer)) {
            continue;
        }
        if (blurLayers.size() > 0 && blurLayers.front() == layer) {
            blurLayers.pop_front();

            auto status = mBlurFilter->prepare();
            if (status != NO_ERROR) {
                ALOGE("Failed to render blur effect! Aborting GPU composition for buffer (%p).",
                      buffer->getBuffer()->handle);
                checkErrors("Can't render first blur pass");
                resultPromise->set_value({status, base::unique_fd()});
                return;
            }

            if (blurLayers.size() == 0) {
                // Done blurring, time to bind the native FBO and render our blur onto it.
                fbo = std::make_unique<BindNativeBufferAsFramebuffer>(*this,
                                                                      buffer.get()
                                                                              ->getBuffer()
                                                                              ->getNativeBuffer(),
                                                                      useFramebufferCache);
                status = fbo->getStatus();
                setViewportAndProjection(display.physicalDisplay, display.clip);
            } else {
                // There's still something else to blur, so let's keep rendering to our FBO
                // instead of to the display.
                status = mBlurFilter->setAsDrawTarget(display,
                                                      blurLayers.front().backgroundBlurRadius);
            }
            if (status != NO_ERROR) {
                ALOGE("Failed to bind framebuffer! Aborting GPU composition for buffer (%p).",
                      buffer->getBuffer()->handle);
                checkErrors("Can't bind native framebuffer");
                resultPromise->set_value({status, base::unique_fd()});
                return;
            }

            status = mBlurFilter->render(blurLayersSize > 1);
            if (status != NO_ERROR) {
                ALOGE("Failed to render blur effect! Aborting GPU composition for buffer (%p).",
                      buffer->getBuffer()->handle);
                checkErrors("Can't render blur filter");
                resultPromise->set_value({status, base::unique_fd()});
                return;
            }
        }

        // Ensure luminance is at least 100 nits to avoid div-by-zero
        const float maxLuminance = std::max(100.f, layer.source.buffer.maxLuminanceNits);
        mState.maxMasteringLuminance = maxLuminance;
        mState.maxContentLuminance = maxLuminance;
        mState.projectionMatrix = projectionMatrix * layer.geometry.positionTransform;

        // PrismOS bring-up diagnostics: once per second, the first flat layers with their
        // composed clip-space translation — reveals layers that land off-screen.
        {
            static uint32_t flatDiagFrame = 0;
            static int flatDiagCount = 0;
            if (flatDiagFrame != mPrismSceneFrame || flatDiagCount < 4) {
                if (flatDiagFrame != mPrismSceneFrame) {
                    flatDiagFrame = mPrismSceneFrame;
                    flatDiagCount = 0;
                }
                if (flatDiagCount < 4) {
                    const float* m = mState.projectionMatrix.asArray();
                    const float ndcX = m[12] / m[15];
                    const float ndcY = m[13] / m[15];
                    ALOGD("PrismOS flat: '%s' skip=%d ndc=(%.2f, %.2f, w=%.2f)", layer.name.c_str(),
                          layer.skipContentDraw ? 1 : 0, ndcX, ndcY, m[15]);
                    flatDiagCount++;
                }
            }
        }

        const FloatRect bounds = layer.geometry.boundaries;
        Mesh::VertexArray<vec2> position(mesh.getPositionArray<vec2>());
        position[0] = vec2(bounds.left, bounds.top);
        position[1] = vec2(bounds.left, bounds.bottom);
        position[2] = vec2(bounds.right, bounds.bottom);
        position[3] = vec2(bounds.right, bounds.top);

        setupLayerCropping(layer, mesh);
        setColorTransform(layer.colorTransform);

        bool usePremultipliedAlpha = true;
        bool disableTexture = true;
        bool isOpaque = false;
        if (layer.source.buffer.buffer != nullptr) {
            disableTexture = false;
            isOpaque = layer.source.buffer.isOpaque;

            sp<GraphicBuffer> gBuf = layer.source.buffer.buffer->getBuffer();
            validateInputBufferUsage(gBuf);
            bindExternalTextureBuffer(layer.source.buffer.textureName, gBuf,
                                      layer.source.buffer.fence);

            usePremultipliedAlpha = layer.source.buffer.usePremultipliedAlpha;
            Texture texture(Texture::TEXTURE_EXTERNAL, layer.source.buffer.textureName);
            mat4 texMatrix = layer.source.buffer.textureTransform;

            texture.setMatrix(texMatrix.asArray());
            texture.setFiltering(layer.source.buffer.useTextureFiltering);

            texture.setDimensions(gBuf->getWidth(), gBuf->getHeight());
            setSourceY410BT2020(layer.source.buffer.isY410BT2020);

            renderengine::Mesh::VertexArray<vec2> texCoords(mesh.getTexCoordArray<vec2>());
            texCoords[0] = vec2(0.0, 0.0);
            texCoords[1] = vec2(0.0, 1.0);
            texCoords[2] = vec2(1.0, 1.0);
            texCoords[3] = vec2(1.0, 0.0);
            setupLayerTexturing(texture);

            // Do not cache protected EGLImage, protected memory is limited.
            if (gBuf->getUsage() & GRALLOC_USAGE_PROTECTED) {
                unmapExternalTextureBuffer(gBuf);
            }
        }

        const half3 solidColor = layer.source.solidColor;
        const half4 color = half4(solidColor.r, solidColor.g, solidColor.b, layer.alpha);
        const float radius =
                (layer.geometry.roundedCornersRadius.x + layer.geometry.roundedCornersRadius.y) /
                2.0f;
        // Buffer sources will have a black solid color ignored in the shader,
        // so in that scenario the solid color passed here is arbitrary.
        setupLayerBlending(usePremultipliedAlpha, isOpaque, disableTexture, color, radius);
        if (layer.disableBlending) {
            glDisable(GL_BLEND);
        }
        setSourceDataSpace(layer.sourceDataspace);

        if (layer.shadow.length > 0.0f) {
            handleShadow(layer.geometry.boundaries, radius, layer.shadow);
        }
        // We only want to do a special handling for rounded corners when having rounded corners
        // is the only reason it needs to turn on blending, otherwise, we handle it like the
        // usual way since it needs to turn on blending anyway.
        else if (radius > 0.0 && color.a >= 1.0f && isOpaque) {
            handleRoundedCorners(display, layer, mesh);
        } else {
            drawMesh(mesh);
        }

        // Cleanup if there's a buffer source
        if (layer.source.buffer.buffer != nullptr) {
            disableBlending();
            setSourceY410BT2020(false);
            disableTexturing();
        }
    }

    // PrismOS: gated dump of the composed flat frame (exactly what the HWC would
    // present) for offline inspection. Rate-limited to one frame per second while
    // the gate is on; pull with `adb pull /data/local/tmp/prism_flat.ppm`.
    if (base::GetBoolProperty("debug.prismos.env.frame_dump", false)) {
        static std::chrono::steady_clock::time_point sLastDump;
        const auto now = std::chrono::steady_clock::now();
        if (now - sLastDump > std::chrono::seconds(1)) {
            sLastDump = now;
            GLint vp[4] = {0, 0, 0, 0};
            glGetIntegerv(GL_VIEWPORT, vp);
            (void)vp;
            writePPM("/data/local/tmp/prism_flat", mVpWidth, mVpHeight);
            ALOGD("PRISM-DUMP: flat frame dumped %ux%u", mVpWidth, mVpHeight);
        }
    }

    base::unique_fd drawFence = flush();

    // If flush failed or we don't support native fences, we need to force the
    // gl command stream to be executed.
    if (drawFence.get() < 0) {
        bool success = finish();
        if (!success) {
            ALOGE("Failed to flush RenderEngine commands");
            checkErrors();
            // Chances are, something illegal happened (either the caller passed
            // us bad parameters, or we messed up our shader generation).
            resultPromise->set_value({INVALID_OPERATION, std::move(drawFence)});
            return;
        }
        mLastDrawFence = nullptr;
    } else {
        // The caller takes ownership of drawFence, so we need to duplicate the
        // fd here.
        mLastDrawFence = new Fence(dup(drawFence.get()));
    }
    mPriorResourcesCleaned = false;

    checkErrors();
    resultPromise->set_value({NO_ERROR, std::move(drawFence)});
    return;
}

void GLESRenderEngine::setViewportAndProjection(Rect viewport, Rect clip) {
    ATRACE_CALL();
    mVpWidth = viewport.getWidth();
    mVpHeight = viewport.getHeight();

    // We pass the the top left corner instead of the bottom left corner,
    // because since we're rendering off-screen first.
    glViewport(viewport.left, viewport.top, mVpWidth, mVpHeight);

    mState.projectionMatrix = mat4::ortho(clip.left, clip.right, clip.top, clip.bottom, 0, 1);
}

void GLESRenderEngine::setupLayerBlending(bool premultipliedAlpha, bool opaque, bool disableTexture,
                                          const half4& color, float cornerRadius) {
    mState.isPremultipliedAlpha = premultipliedAlpha;
    mState.isOpaque = opaque;
    mState.color = color;
    mState.cornerRadius = cornerRadius;

    if (disableTexture) {
        mState.textureEnabled = false;
    }

    if (color.a < 1.0f || !opaque || cornerRadius > 0.0f) {
        glEnable(GL_BLEND);
        glBlendFuncSeparate(premultipliedAlpha ? GL_ONE : GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                            GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    } else {
        glDisable(GL_BLEND);
    }
}

void GLESRenderEngine::setSourceY410BT2020(bool enable) {
    mState.isY410BT2020 = enable;
}

void GLESRenderEngine::setSourceDataSpace(Dataspace source) {
    mDataSpace = source;
}

void GLESRenderEngine::setOutputDataSpace(Dataspace dataspace) {
    mOutputDataSpace = dataspace;
}

void GLESRenderEngine::setDisplayMaxLuminance(const float maxLuminance) {
    mState.displayMaxLuminance = maxLuminance;
}

void GLESRenderEngine::setupLayerTexturing(const Texture& texture) {
    GLuint target = texture.getTextureTarget();
    glBindTexture(target, texture.getTextureName());
    GLenum filter = GL_NEAREST;
    if (texture.getFiltering()) {
        filter = GL_LINEAR;
    }
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter);

    mState.texture = texture;
    mState.textureEnabled = true;
}

void GLESRenderEngine::setColorTransform(const mat4& colorTransform) {
    mState.colorMatrix = colorTransform;
}

void GLESRenderEngine::setDisplayColorTransform(const mat4& colorTransform) {
    mState.displayColorMatrix = colorTransform;
}

void GLESRenderEngine::disableTexturing() {
    mState.textureEnabled = false;
}

void GLESRenderEngine::disableBlending() {
    glDisable(GL_BLEND);
}

void GLESRenderEngine::setupFillWithColor(float r, float g, float b, float a) {
    mState.isPremultipliedAlpha = true;
    mState.isOpaque = false;
    mState.color = half4(r, g, b, a);
    mState.textureEnabled = false;
    glDisable(GL_BLEND);
}

void GLESRenderEngine::setupCornerRadiusCropSize(float width, float height) {
    mState.cropSize = half2(width, height);
}

void GLESRenderEngine::drawMesh(const Mesh& mesh) {
    ATRACE_CALL();
    if (mesh.getTexCoordsSize()) {
        glEnableVertexAttribArray(Program::texCoords);
        glVertexAttribPointer(Program::texCoords, mesh.getTexCoordsSize(), GL_FLOAT, GL_FALSE,
                              mesh.getByteStride(), mesh.getTexCoords());
    }

    // PrismOS: the position array must be enabled explicitly. The upstream GLES path never
    // enables it (dead code once Skia became the default backend), so with the GLES backend
    // selected via debug.renderengine.backend every drawMesh rasterizes a degenerate triangle
    // — silently black captures and invisible client-composited windows.
    glEnableVertexAttribArray(Program::position);
    glVertexAttribPointer(Program::position, mesh.getVertexSize(), GL_FLOAT, GL_FALSE,
                          mesh.getByteStride(), mesh.getPositions());

    if (mState.cornerRadius > 0.0f) {
        glEnableVertexAttribArray(Program::cropCoords);
        glVertexAttribPointer(Program::cropCoords, mesh.getVertexSize(), GL_FLOAT, GL_FALSE,
                              mesh.getByteStride(), mesh.getCropCoords());
    }

    if (mState.drawShadows) {
        glEnableVertexAttribArray(Program::shadowColor);
        glVertexAttribPointer(Program::shadowColor, mesh.getShadowColorSize(), GL_FLOAT, GL_FALSE,
                              mesh.getByteStride(), mesh.getShadowColor());

        glEnableVertexAttribArray(Program::shadowParams);
        glVertexAttribPointer(Program::shadowParams, mesh.getShadowParamsSize(), GL_FLOAT, GL_FALSE,
                              mesh.getByteStride(), mesh.getShadowParams());
    }

    Description managedState = mState;
    // By default, DISPLAY_P3 is the only supported wide color output. However,
    // when HDR content is present, hardware composer may be able to handle
    // BT2020 data space, in that case, the output data space is set to be
    // BT2020_HLG or BT2020_PQ respectively. In GPU fall back we need
    // to respect this and convert non-HDR content to HDR format.
    if (mUseColorManagement) {
        Dataspace inputStandard = static_cast<Dataspace>(mDataSpace & Dataspace::STANDARD_MASK);
        Dataspace inputTransfer = static_cast<Dataspace>(mDataSpace & Dataspace::TRANSFER_MASK);
        Dataspace outputStandard =
                static_cast<Dataspace>(mOutputDataSpace & Dataspace::STANDARD_MASK);
        Dataspace outputTransfer =
                static_cast<Dataspace>(mOutputDataSpace & Dataspace::TRANSFER_MASK);
        bool needsXYZConversion = needsXYZTransformMatrix();

        // NOTE: if the input standard of the input dataspace is not STANDARD_DCI_P3 or
        // STANDARD_BT2020, it will be  treated as STANDARD_BT709
        if (inputStandard != Dataspace::STANDARD_DCI_P3 &&
            inputStandard != Dataspace::STANDARD_BT2020) {
            inputStandard = Dataspace::STANDARD_BT709;
        }

        if (needsXYZConversion) {
            // The supported input color spaces are standard RGB, Display P3 and BT2020.
            switch (inputStandard) {
                case Dataspace::STANDARD_DCI_P3:
                    managedState.inputTransformMatrix = mDisplayP3ToXyz;
                    break;
                case Dataspace::STANDARD_BT2020:
                    managedState.inputTransformMatrix = mBt2020ToXyz;
                    break;
                default:
                    managedState.inputTransformMatrix = mSrgbToXyz;
                    break;
            }

            // The supported output color spaces are BT2020, Display P3 and standard RGB.
            switch (outputStandard) {
                case Dataspace::STANDARD_BT2020:
                    managedState.outputTransformMatrix = mXyzToBt2020;
                    break;
                case Dataspace::STANDARD_DCI_P3:
                    managedState.outputTransformMatrix = mXyzToDisplayP3;
                    break;
                default:
                    managedState.outputTransformMatrix = mXyzToSrgb;
                    break;
            }
        } else if (inputStandard != outputStandard) {
            // At this point, the input data space and output data space could be both
            // HDR data spaces, but they match each other, we do nothing in this case.
            // In addition to the case above, the input data space could be
            // - scRGB linear
            // - scRGB non-linear
            // - sRGB
            // - Display P3
            // - BT2020
            // The output data spaces could be
            // - sRGB
            // - Display P3
            // - BT2020
            switch (outputStandard) {
                case Dataspace::STANDARD_BT2020:
                    if (inputStandard == Dataspace::STANDARD_BT709) {
                        managedState.outputTransformMatrix = mSrgbToBt2020;
                    } else if (inputStandard == Dataspace::STANDARD_DCI_P3) {
                        managedState.outputTransformMatrix = mDisplayP3ToBt2020;
                    }
                    break;
                case Dataspace::STANDARD_DCI_P3:
                    if (inputStandard == Dataspace::STANDARD_BT709) {
                        managedState.outputTransformMatrix = mSrgbToDisplayP3;
                    } else if (inputStandard == Dataspace::STANDARD_BT2020) {
                        managedState.outputTransformMatrix = mBt2020ToDisplayP3;
                    }
                    break;
                default:
                    if (inputStandard == Dataspace::STANDARD_DCI_P3) {
                        managedState.outputTransformMatrix = mDisplayP3ToSrgb;
                    } else if (inputStandard == Dataspace::STANDARD_BT2020) {
                        managedState.outputTransformMatrix = mBt2020ToSrgb;
                    }
                    break;
            }
        }

        // we need to convert the RGB value to linear space and convert it back when:
        // - there is a color matrix that is not an identity matrix, or
        // - there is an output transform matrix that is not an identity matrix, or
        // - the input transfer function doesn't match the output transfer function.
        if (managedState.hasColorMatrix() || managedState.hasOutputTransformMatrix() ||
            inputTransfer != outputTransfer) {
            managedState.inputTransferFunction =
                    Description::dataSpaceToTransferFunction(inputTransfer);
            managedState.outputTransferFunction =
                    Description::dataSpaceToTransferFunction(outputTransfer);
        }
    }

    ProgramCache::getInstance().useProgram(mInProtectedContext ? mProtectedEGLContext : mEGLContext,
                                           managedState);

    if (mState.drawShadows) {
        glDrawElements(mesh.getPrimitive(), mesh.getIndexCount(), GL_UNSIGNED_SHORT,
                       mesh.getIndices());
    } else {
        glDrawArrays(mesh.getPrimitive(), 0, mesh.getVertexCount());
    }

    if (mUseColorManagement && outputDebugPPMs) {
        static uint64_t managedColorFrameCount = 0;
        std::ostringstream out;
        out << "/data/texture_out" << managedColorFrameCount++;
        writePPM(out.str().c_str(), mVpWidth, mVpHeight);
    }

    if (mesh.getTexCoordsSize()) {
        glDisableVertexAttribArray(Program::texCoords);
    }

    if (mState.cornerRadius > 0.0f) {
        glDisableVertexAttribArray(Program::cropCoords);
    }

    if (mState.drawShadows) {
        glDisableVertexAttribArray(Program::shadowColor);
        glDisableVertexAttribArray(Program::shadowParams);
    }
}

size_t GLESRenderEngine::getMaxTextureSize() const {
    return mMaxTextureSize;
}

size_t GLESRenderEngine::getMaxViewportDims() const {
    return mMaxViewportDims[0] < mMaxViewportDims[1] ? mMaxViewportDims[0] : mMaxViewportDims[1];
}

void GLESRenderEngine::dump(std::string& result) {
    const GLExtensions& extensions = GLExtensions::getInstance();
    ProgramCache& cache = ProgramCache::getInstance();

    StringAppendF(&result, "EGL implementation : %s\n", extensions.getEGLVersion());
    StringAppendF(&result, "%s\n", extensions.getEGLExtensions());
    StringAppendF(&result, "GLES: %s, %s, %s\n", extensions.getVendor(), extensions.getRenderer(),
                  extensions.getVersion());
    StringAppendF(&result, "%s\n", extensions.getExtensions());
    StringAppendF(&result, "RenderEngine supports protected context: %d\n",
                  supportsProtectedContent());
    StringAppendF(&result, "RenderEngine is in protected context: %d\n", mInProtectedContext);
    StringAppendF(&result, "RenderEngine program cache size for unprotected context: %zu\n",
                  cache.getSize(mEGLContext));
    StringAppendF(&result, "RenderEngine program cache size for protected context: %zu\n",
                  cache.getSize(mProtectedEGLContext));
    StringAppendF(&result, "RenderEngine last dataspace conversion: (%s) to (%s)\n",
                  dataspaceDetails(static_cast<android_dataspace>(mDataSpace)).c_str(),
                  dataspaceDetails(static_cast<android_dataspace>(mOutputDataSpace)).c_str());
    {
        std::lock_guard<std::mutex> lock(mRenderingMutex);
        StringAppendF(&result, "RenderEngine image cache size: %zu\n", mImageCache.size());
        StringAppendF(&result, "Dumping buffer ids...\n");
        for (const auto& [id, unused] : mImageCache) {
            StringAppendF(&result, "0x%" PRIx64 "\n", id);
        }
    }
    {
        std::lock_guard<std::mutex> lock(mFramebufferImageCacheMutex);
        StringAppendF(&result, "RenderEngine framebuffer image cache size: %zu\n",
                      mFramebufferImageCache.size());
        StringAppendF(&result, "Dumping buffer ids...\n");
        for (const auto& [id, unused] : mFramebufferImageCache) {
            StringAppendF(&result, "0x%" PRIx64 "\n", id);
        }
    }
}

GLESRenderEngine::GlesVersion GLESRenderEngine::parseGlesVersion(const char* str) {
    int major, minor;
    if (sscanf(str, "OpenGL ES-CM %d.%d", &major, &minor) != 2) {
        if (sscanf(str, "OpenGL ES %d.%d", &major, &minor) != 2) {
            ALOGW("Unable to parse GL_VERSION string: \"%s\"", str);
            return GLES_VERSION_1_0;
        }
    }

    if (major == 1 && minor == 0) return GLES_VERSION_1_0;
    if (major == 1 && minor >= 1) return GLES_VERSION_1_1;
    if (major == 2 && minor >= 0) return GLES_VERSION_2_0;
    if (major == 3 && minor >= 0) return GLES_VERSION_3_0;

    ALOGW("Unrecognized OpenGL ES version: %d.%d", major, minor);
    return GLES_VERSION_1_0;
}

EGLContext GLESRenderEngine::createEglContext(EGLDisplay display, EGLConfig config,
                                              EGLContext shareContext,
                                              std::optional<ContextPriority> contextPriority,
                                              Protection protection) {
    EGLint renderableType = 0;
    if (config == EGL_NO_CONFIG) {
        renderableType = EGL_OPENGL_ES3_BIT;
    } else if (!eglGetConfigAttrib(display, config, EGL_RENDERABLE_TYPE, &renderableType)) {
        LOG_ALWAYS_FATAL("can't query EGLConfig RENDERABLE_TYPE");
    }
    EGLint contextClientVersion = 0;
    if (renderableType & EGL_OPENGL_ES3_BIT) {
        contextClientVersion = 3;
    } else if (renderableType & EGL_OPENGL_ES2_BIT) {
        contextClientVersion = 2;
    } else if (renderableType & EGL_OPENGL_ES_BIT) {
        contextClientVersion = 1;
    } else {
        LOG_ALWAYS_FATAL("no supported EGL_RENDERABLE_TYPEs");
    }

    std::vector<EGLint> contextAttributes;
    contextAttributes.reserve(7);
    contextAttributes.push_back(EGL_CONTEXT_CLIENT_VERSION);
    contextAttributes.push_back(contextClientVersion);
    if (contextPriority) {
        contextAttributes.push_back(EGL_CONTEXT_PRIORITY_LEVEL_IMG);
        switch (*contextPriority) {
            case ContextPriority::REALTIME:
                contextAttributes.push_back(EGL_CONTEXT_PRIORITY_REALTIME_NV);
                break;
            case ContextPriority::MEDIUM:
                contextAttributes.push_back(EGL_CONTEXT_PRIORITY_MEDIUM_IMG);
                break;
            case ContextPriority::LOW:
                contextAttributes.push_back(EGL_CONTEXT_PRIORITY_LOW_IMG);
                break;
            case ContextPriority::HIGH:
            default:
                contextAttributes.push_back(EGL_CONTEXT_PRIORITY_HIGH_IMG);
                break;
        }
    }
    if (protection == Protection::PROTECTED) {
        contextAttributes.push_back(EGL_PROTECTED_CONTENT_EXT);
        contextAttributes.push_back(EGL_TRUE);
    }
    contextAttributes.push_back(EGL_NONE);

    EGLContext context = eglCreateContext(display, config, shareContext, contextAttributes.data());

    if (contextClientVersion == 3 && context == EGL_NO_CONTEXT) {
        // eglGetConfigAttrib indicated we can create GLES 3 context, but we failed, thus
        // EGL_NO_CONTEXT so that we can abort.
        if (config != EGL_NO_CONFIG) {
            return context;
        }
        // If |config| is EGL_NO_CONFIG, we speculatively try to create GLES 3 context, so we should
        // try to fall back to GLES 2.
        contextAttributes[1] = 2;
        context = eglCreateContext(display, config, shareContext, contextAttributes.data());
    }

    return context;
}

EGLSurface GLESRenderEngine::createStubEglPbufferSurface(EGLDisplay display, EGLConfig config,
                                                         int hwcFormat, Protection protection) {
    EGLConfig stubConfig = config;
    if (stubConfig == EGL_NO_CONFIG) {
        stubConfig = chooseEglConfig(display, hwcFormat, /*logConfig*/ true);
    }
    std::vector<EGLint> attributes;
    attributes.reserve(7);
    attributes.push_back(EGL_WIDTH);
    attributes.push_back(1);
    attributes.push_back(EGL_HEIGHT);
    attributes.push_back(1);
    if (protection == Protection::PROTECTED) {
        attributes.push_back(EGL_PROTECTED_CONTENT_EXT);
        attributes.push_back(EGL_TRUE);
    }
    attributes.push_back(EGL_NONE);

    return eglCreatePbufferSurface(display, stubConfig, attributes.data());
}

bool GLESRenderEngine::isHdrDataSpace(const Dataspace dataSpace) const {
    const Dataspace standard = static_cast<Dataspace>(dataSpace & Dataspace::STANDARD_MASK);
    const Dataspace transfer = static_cast<Dataspace>(dataSpace & Dataspace::TRANSFER_MASK);
    return standard == Dataspace::STANDARD_BT2020 &&
            (transfer == Dataspace::TRANSFER_ST2084 || transfer == Dataspace::TRANSFER_HLG);
}

// For convenience, we want to convert the input color space to XYZ color space first,
// and then convert from XYZ color space to output color space when
// - SDR and HDR contents are mixed, either SDR content will be converted to HDR or
//   HDR content will be tone-mapped to SDR; Or,
// - there are HDR PQ and HLG contents presented at the same time, where we want to convert
//   HLG content to PQ content.
// In either case above, we need to operate the Y value in XYZ color space. Thus, when either
// input data space or output data space is HDR data space, and the input transfer function
// doesn't match the output transfer function, we would enable an intermediate transfrom to
// XYZ color space.
bool GLESRenderEngine::needsXYZTransformMatrix() const {
    const bool isInputHdrDataSpace = isHdrDataSpace(mDataSpace);
    const bool isOutputHdrDataSpace = isHdrDataSpace(mOutputDataSpace);
    const Dataspace inputTransfer = static_cast<Dataspace>(mDataSpace & Dataspace::TRANSFER_MASK);
    const Dataspace outputTransfer =
            static_cast<Dataspace>(mOutputDataSpace & Dataspace::TRANSFER_MASK);

    return (isInputHdrDataSpace || isOutputHdrDataSpace) && inputTransfer != outputTransfer;
}

bool GLESRenderEngine::isImageCachedForTesting(uint64_t bufferId) {
    std::lock_guard<std::mutex> lock(mRenderingMutex);
    const auto& cachedImage = mImageCache.find(bufferId);
    return cachedImage != mImageCache.end();
}

bool GLESRenderEngine::isTextureNameKnownForTesting(uint32_t texName) {
    const auto& entry = mTextureView.find(texName);
    return entry != mTextureView.end();
}

std::optional<uint64_t> GLESRenderEngine::getBufferIdForTextureNameForTesting(uint32_t texName) {
    const auto& entry = mTextureView.find(texName);
    return entry != mTextureView.end() ? entry->second : std::nullopt;
}

bool GLESRenderEngine::isFramebufferImageCachedForTesting(uint64_t bufferId) {
    std::lock_guard<std::mutex> lock(mFramebufferImageCacheMutex);
    return std::any_of(mFramebufferImageCache.cbegin(), mFramebufferImageCache.cend(),
                       [=](std::pair<uint64_t, EGLImageKHR> image) {
                           return image.first == bufferId;
                       });
}

// FlushTracer implementation
GLESRenderEngine::FlushTracer::FlushTracer(GLESRenderEngine* engine) : mEngine(engine) {
    mThread = std::thread(&GLESRenderEngine::FlushTracer::loop, this);
}

GLESRenderEngine::FlushTracer::~FlushTracer() {
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mRunning = false;
    }
    mCondition.notify_all();
    if (mThread.joinable()) {
        mThread.join();
    }
}

void GLESRenderEngine::FlushTracer::queueSync(EGLSyncKHR sync) {
    std::lock_guard<std::mutex> lock(mMutex);
    char name[64];
    const uint64_t frameNum = mFramesQueued++;
    snprintf(name, sizeof(name), "Queueing sync for frame: %lu",
             static_cast<unsigned long>(frameNum));
    ATRACE_NAME(name);
    mQueue.push({sync, frameNum});
    ATRACE_INT("GPU Frames Outstanding", mQueue.size());
    mCondition.notify_one();
}

void GLESRenderEngine::FlushTracer::loop() {
    while (mRunning) {
        QueueEntry entry;
        {
            std::lock_guard<std::mutex> lock(mMutex);

            mCondition.wait(mMutex,
                            [&]() REQUIRES(mMutex) { return !mQueue.empty() || !mRunning; });

            if (!mRunning) {
                // if mRunning is false, then FlushTracer is being destroyed, so
                // bail out now.
                break;
            }
            entry = mQueue.front();
            mQueue.pop();
        }
        {
            char name[64];
            snprintf(name, sizeof(name), "waiting for frame %lu",
                     static_cast<unsigned long>(entry.mFrameNum));
            ATRACE_NAME(name);
            mEngine->waitSync(entry.mSync, 0);
        }
    }
}

void GLESRenderEngine::handleShadow(const FloatRect& casterRect, float casterCornerRadius,
                                    const ShadowSettings& settings) {
    ATRACE_CALL();
    const float casterZ = settings.length / 2.0f;
    const GLShadowVertexGenerator shadows(casterRect, casterCornerRadius, casterZ,
                                          settings.casterIsTranslucent, settings.ambientColor,
                                          settings.spotColor, settings.lightPos,
                                          settings.lightRadius);

    // setup mesh for both shadows
    Mesh mesh = Mesh::Builder()
                        .setPrimitive(Mesh::TRIANGLES)
                        .setVertices(shadows.getVertexCount(), 2 /* size */)
                        .setShadowAttrs()
                        .setIndices(shadows.getIndexCount())
                        .build();

    Mesh::VertexArray<vec2> position = mesh.getPositionArray<vec2>();
    Mesh::VertexArray<vec4> shadowColor = mesh.getShadowColorArray<vec4>();
    Mesh::VertexArray<vec3> shadowParams = mesh.getShadowParamsArray<vec3>();
    shadows.fillVertices(position, shadowColor, shadowParams);
    shadows.fillIndices(mesh.getIndicesArray());

    mState.cornerRadius = 0.0f;
    mState.drawShadows = true;
    setupLayerTexturing(mShadowTexture->getTexture());
    drawMesh(mesh);
    mState.drawShadows = false;
}

} // namespace gl
} // namespace renderengine
} // namespace android
