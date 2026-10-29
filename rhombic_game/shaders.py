"""GLSL for the three renderers.  Each renderer has its own math:

- sky:    direction-only.  Positions are directions submitted with
          w = 0; colour is a function of direction alone.
- planes: homogeneous positions.  The fragment shader recovers the
          world point as xyz / w, exact at every distance.
- blocks: the one ordinary finite renderer, for local geometry --
          but still projected with the infinite projection matrix.
"""

SUN_DIR = (-0.45, 0.62, -0.35)  # static daytime sun (normalised in shader)
# One horizon colour, shared by everything that reaches the horizon:
# the sky gradient's horizon value (in SKY_FRAG), the planes' distance
# haze, and the GL clear colour.  If these disagree, the seam between
# a hazed-out plane and the sky shows as a grey band at the horizon.
HAZE_COLOR = (0.78, 0.86, 0.94)

SKY_VERT = """
#version 330 core
layout(location = 0) in vec3 aDir;
uniform mat4 uVP;
out vec3 vDir;
void main() {
    vDir = aDir;
    // w = 0: the view translation drops out by itself, and the
    // infinite projection lands this exactly on the far plane.
    gl_Position = uVP * vec4(aDir, 0.0);
}
"""

SKY_FRAG = """
#version 330 core
in vec3 vDir;
uniform vec3 uSunDir;
out vec4 fragColor;
void main() {
    vec3 d = normalize(vDir);
    vec3 zenith = vec3(0.22, 0.46, 0.82);
    vec3 horizon = vec3(0.78, 0.86, 0.94);
    vec3 below = vec3(0.55, 0.60, 0.64);
    vec3 col = d.y >= 0.0
        ? mix(horizon, zenith, pow(d.y, 0.55))
        : mix(horizon, below, clamp(-d.y * 3.0, 0.0, 1.0));
    float s = max(dot(d, uSunDir), 0.0);
    col += vec3(1.0, 0.95, 0.85) * smoothstep(0.9993, 0.9997, s) * 1.6; // disc
    col += vec3(1.0, 0.97, 0.88) * pow(s, 350.0) * 0.35;              // corona
    col += vec3(0.9, 0.95, 1.0) * pow(s, 8.0) * 0.10;                 // wide glow
    fragColor = vec4(col, 1.0);
}
"""

PLANE_VERT = """
#version 330 core
layout(location = 0) in vec4 aHomog;
uniform mat4 uVP;
out vec4 vHomog;
void main() {
    vHomog = aHomog;
    gl_Position = uVP * aHomog;
}
"""

_FILTERED_CHECKER = """
// Box-filtered checkerboard over the fragment's actual footprint.
// The checker factors as s(x) * s(y) with s a +-1 square wave, and the
// box average of a product over a product region is the product of the
// box averages -- so filter each axis over the footprint's extent in
// that axis.  The extents come from the screen-space derivatives of
// the pattern coordinates, i.e. the (anisotropic) footprint itself,
// approximated by its bounding box.  At grazing angles the across-view
// cells therefore survive as stripes long after a scalar fwidth fade
// would have averaged the whole pattern to grey.
float _tri(float t) { return 1.0 - abs(1.0 - mod(t, 2.0)); }
float _filteredAxis(float t, float w) {
    w = max(w, 1e-4);
    return (_tri(t + 0.5 * w) - _tri(t - 0.5 * w)) / w;
}
float filteredChecker(vec2 g) {
    vec2 dx = dFdx(g);
    vec2 dy = dFdy(g);
    // Widths inflated by 1.25: bias the filter toward stability --
    // the moire band lives where the footprint spans ~1-2 cells, and
    // a slightly wider box costs little sharpness there.
    float fx = _filteredAxis(g.x, (abs(dx.x) + abs(dy.x)) * 1.25);
    float fy = _filteredAxis(g.y, (abs(dx.y) + abs(dy.y)) * 1.25);
    return 0.5 + 0.5 * fx * fy;
}

// Grid lines must vanish once their cells go sub-pixel: a line
// pattern sampled below its own period aliases into moire bands no
// matter how it is faded by distance, so gate by footprint instead.
float gridFade(vec2 g) {
    vec2 fw = fwidth(g);
    return 1.0 - smoothstep(0.30, 0.60, max(fw.x, fw.y));
}
"""

_BEDROCK_FRAG = (
    """
#version 330 core
in vec4 vHomog;
uniform vec3 uCamPos;
uniform vec3 uHazeColor;
out vec4 fragColor;

float gridLine(vec2 g) {
    vec2 fw = fwidth(g) + vec2(1e-4);
    vec2 a = abs(fract(g - 0.5) - 0.5) / fw;
    return 1.0 - min(min(a.x, a.y), 1.0);
}
"""
    + _FILTERED_CHECKER
    + """
void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    // 8 m checkerboard, footprint-filtered (see filteredChecker); the
    // pattern is what makes fast flight over the bedrock plane visible.
    vec2 g8 = wp.xz / 8.0;
    float check = filteredChecker(g8);
    vec3 col = mix(vec3(0.30, 0.28, 0.26), vec3(0.56, 0.53, 0.49), check);

    float line1 = gridLine(wp.xz) * gridFade(wp.xz);
    float line8 = gridLine(g8) * gridFade(g8) * (1.0 - smoothstep(400.0, 3000.0, dist));
    col = mix(col, vec3(0.16, 0.15, 0.14), line1 * 0.25);
    col = mix(col, vec3(0.10, 0.10, 0.10), line8 * 0.50);

    col = mix(col, uHazeColor, smoothstep(800.0, 7000.0, dist));
    fragColor = vec4(col, 1.0);
}
"""
)

_SEA_FRAG = (
    """
#version 330 core
in vec4 vHomog;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uHazeColor;
out vec4 fragColor;
"""
    + _FILTERED_CHECKER
    + """
void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    float check = filteredChecker(wp.xz / 25.0);
    vec3 col = mix(vec3(0.045, 0.21, 0.37), vec3(0.075, 0.28, 0.45), check * 0.6);

    // Sun glitter on a flat sea: Blinn specular against the up normal.
    vec3 viewDir = normalize(wp - uCamPos);
    vec3 halfVec = normalize(uSunDir - viewDir);
    float spec = pow(max(dot(vec3(0.0, 1.0, 0.0), halfVec), 0.0), 240.0);
    col += vec3(1.0, 0.97, 0.88) * spec * 1.1;
    float sheen = pow(max(dot(vec3(0.0, 1.0, 0.0), halfVec), 0.0), 8.0);
    col += vec3(0.5, 0.6, 0.7) * sheen * 0.08;

    col = mix(col, uHazeColor, smoothstep(600.0, 6000.0, dist));
    fragColor = vec4(col, 0.86);
}
"""
)

PLANE_FRAGS = {"bedrock": _BEDROCK_FRAG, "sea": _SEA_FRAG}

BLOCK_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormalFlat;
layout(location = 2) in vec3 aNormalSmooth;
layout(location = 3) in vec3 aColor;
uniform mat4 uVP;
uniform float uSmooth;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;
void main() {
    vNormal = mix(aNormalFlat, aNormalSmooth, uSmooth);
    vColor = aColor;
    vWorld = aPos;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

BLOCK_FRAG = """
#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
uniform vec3 uSunDir;
uniform vec3 uCamPos;
uniform vec3 uHazeColor;
out vec4 fragColor;
void main() {
    float ndl = max(dot(normalize(vNormal), uSunDir), 0.0);
    vec3 col = vColor * (0.38 + 0.62 * ndl);
    float dist = length(vWorld - uCamPos);
    col = mix(col, uHazeColor, smoothstep(900.0, 7000.0, dist));
    fragColor = vec4(col, 1.0);
}
"""
