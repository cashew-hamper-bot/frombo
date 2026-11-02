"""GLSL for the three renderers.  Each renderer has its own math:

- sky:    direction-only.  Positions are directions submitted with
          w = 0; colour is a function of direction alone.
- planes: homogeneous positions.  The fragment shader recovers the
          world point as xyz / w, exact at every distance.
- blocks: the one ordinary finite renderer, for local geometry.
          The scene renders in two depth passes (near/far split
          at 300 m); each pass has its own projection.
"""

SUN_DIR = (-0.45, 0.62, -0.35)  # static daytime sun (normalised in shader)

# Sky colours.  Distance haze (restored 2026-11-02 at
# Christopher's direction, reversing the earlier no-haze rule):
# an exponential approach to the sky's horizon colour, capped
# at 90% so the far field never quite dissolves into the sky.
# The haze factor is a pure function of true view distance in
# every shader, so it agrees across the near/far depth split.
HAZE_COLOR = (0.78, 0.86, 0.94)  # the sky horizon colour; also the haze colour
NETHER_HAZE_COLOR = (0.40, 0.11, 0.04)  # the nether horizon colour below bedrock
HAZE_MAX = 0.90
HAZE_SCALE = 3500.0  # metres; the exponential approach scale

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
uniform float uUnder;
out vec4 fragColor;
void main() {
    vec3 d = normalize(vDir);
    // Day palette: below the horizon the sky is deep ocean, darker
    // -- as if you were under the water yourself.
    vec3 zenith = vec3(0.22, 0.46, 0.82);
    vec3 horizon = vec3(0.78, 0.86, 0.94);
    vec3 below = vec3(0.03, 0.12, 0.22);
    // Nether palette (camera below the bedrock): molten dark reds.
    vec3 nZenith = vec3(0.10, 0.02, 0.015);
    vec3 nHorizon = vec3(0.40, 0.11, 0.04);
    vec3 nBelow = vec3(0.05, 0.01, 0.008);
    zenith = mix(zenith, nZenith, uUnder);
    horizon = mix(horizon, nHorizon, uUnder);
    below = mix(below, nBelow, uUnder);
    vec3 col = d.y >= 0.0
        ? mix(horizon, zenith, pow(clamp(d.y, 0.0, 1.0), 0.40))
        : mix(horizon, below, clamp(-d.y * 3.0, 0.0, 1.0));
    float s = max(dot(d, uSunDir), 0.0);
    // The sun is a hexagon (flat top and bottom, points east and
    // west), drawn by its signed distance function in the tangent
    // plane at the sun direction.  Note the flat lower edge is what
    // sits parallel to the horizon at sunrise/sunset; swapping the
    // two SDF terms would put points up/down instead (flat sides
    // east/west).
    vec3 uAxis = normalize(cross(vec3(0.0, 1.0, 0.0), uSunDir)); // ~east-west
    vec3 vAxis = cross(uSunDir, uAxis);                            // ~up
    vec2 tp = vec2(dot(d, uAxis), dot(d, vAxis));
    vec2 ap = abs(tp);
    float hexSdf = max(dot(ap, vec2(0.8660254, 0.5)), ap.y) - 0.052;
    float sunAmt = 1.0 - 0.8 * uUnder;
    col += vec3(1.0, 0.95, 0.85) * (1.0 - smoothstep(-0.002, 0.002, hexSdf)) * 1.6 * sunAmt;
    col += vec3(1.0, 0.97, 0.88) * pow(s, 350.0) * 0.35 * sunAmt;     // corona
    col += vec3(0.9, 0.95, 1.0) * pow(s, 8.0) * 0.10 * sunAmt;        // wide glow
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

_BEDROCK_FRAG = """
#version 330 core
in vec4 vHomog;
uniform vec3 uCamPos;
uniform sampler2D uTex;
uniform sampler2D uNoise;
uniform vec3 uHazeColor;
out vec4 fragColor;

void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    // Hexagonal pattern from a mipmapped texture (rectangular tile,
    // hexagonal symmetry: small hexes 3-coloured, grouped into big
    // hexagons of 49-ish small ones), sampled with explicit
    // gradients so the hardware's anisotropic filtering integrates
    // the true elongated footprint at grazing angles.
    vec2 uv = wp.xz / vec2(42.0, 24.2487113059);
    vec3 col = textureGrad(uTex, uv, dFdx(uv), dFdy(uv)).rgb;

    // Large-scale variation (500 m tileable noise), fading in with
    // distance so the far field has structure like a real surface.
    float n = texture(uNoise, wp.xz / 500.0).r;
    col *= mix(1.0, 0.88 + 0.24 * n, smoothstep(60.0, 600.0, dist));

    // Distance haze, capped at 90% (see the block shader).
    col = mix(col, uHazeColor, 0.90 * (1.0 - exp(-dist / 3500.0)));
    fragColor = vec4(col, 1.0);
}
"""

_SEA_FRAG = (
    """
#version 330 core
in vec4 vHomog;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform sampler2D uTex;
uniform sampler2D uNoise;
uniform float uHorizonAlpha;
uniform vec3 uHazeColor;
out vec4 fragColor;
"""
    + """
void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    // The sea is hexagonal too: the shared pattern texture's alpha
    // holds the hex factor, sampled here at twice the bedrock tile
    // (84 m), so the sea's small hexes are 8 m and its big hexagons
    // ~48 m across.
    vec2 uv = wp.xz / vec2(84.0, 48.4974226119);
    float hexf = textureGrad(uTex, uv, dFdx(uv), dFdy(uv)).a;
    vec3 col = mix(vec3(0.045, 0.21, 0.37), vec3(0.10, 0.34, 0.52), hexf);

    // Large-scale colour drift toward the horizon (500 m tileable
    // noise): the way the open ocean's colour varies when you look
    // out to sea.  Fades in with distance.
    float n = texture(uNoise, wp.xz / 500.0).r;
    vec3 drift = mix(vec3(0.030, 0.16, 0.30), vec3(0.10, 0.32, 0.48), n);
    col = mix(col, drift, smoothstep(120.0, 1200.0, dist) * 0.6);

    // Sun glitter on a flat sea: Blinn specular against the up normal.
    vec3 viewDir = normalize(wp - uCamPos);
    vec3 halfVec = normalize(uSunDir - viewDir);
    float spec = pow(max(dot(vec3(0.0, 1.0, 0.0), halfVec), 0.0), 240.0);
    col += vec3(1.0, 0.97, 0.88) * spec * 1.1;
    float sheen = pow(max(dot(vec3(0.0, 1.0, 0.0), halfVec), 0.0), 8.0);
    col += vec3(0.5, 0.6, 0.7) * sheen * 0.08;

    // Colour haze, capped at 90% (see the block shader).
    col = mix(col, uHazeColor, 0.90 * (1.0 - exp(-dist / 3500.0)));
    // Alpha carries statistics at the horizon: the sea is
    // the transparent partner there, so its opacity fades to
    // uHorizonAlpha = sea_fraction x base opacity -- equivalently
    // horizon transparency = land_fraction + sea_fraction x
    // base transparency.  (Alpha mixture, not colour haze.)
    float hf = 1.0 - exp(-dist * (1.0 / 30000.0));
    fragColor = vec4(col, mix(0.86, uHorizonAlpha, hf));
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
layout(location = 4) in float aAO;
uniform mat4 uVP;
uniform float uSmooth;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;
out float vAO;
void main() {
    vNormal = mix(aNormalFlat, aNormalSmooth, uSmooth);
    vColor = aColor;
    vWorld = aPos;
    vAO = aAO;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

BLOCK_FRAG = """
#version 330 core
in vec3 vNormal;
in vec3 vColor;
in vec3 vWorld;
in float vAO;
uniform vec3 uSunDir;
uniform vec3 uCamPos;
uniform vec3 uHazeColor;
out vec4 fragColor;
void main() {
    vec3 n = normalize(vNormal);
    // Hemisphere ambient: sky blue from above, warm ground bounce
    // from below, blended by the normal's height.
    vec3 hemi = mix(vec3(0.34, 0.29, 0.24), vec3(0.50, 0.64, 0.88), 0.5 + 0.5 * n.y);
    // Sun: a warm point light (at infinity -- the theme's sun).
    float ndl = max(dot(n, uSunDir), 0.0);
    vec3 sun = vec3(1.0, 0.93, 0.80) * ndl;
    // Local ambient occlusion (per-vertex, from neighbour blocks)
    // scales the ambient only -- direct sun still reaches in.  A
    // flat minimum ambient term keeps the darkest vertices readable
    // (no global illumination is coming to save them).
    vec3 col = vColor * (hemi * (0.45 + 0.55 * vAO) + vec3(0.16) + sun * 0.95);
    // Diffuse only, evaluated per pixel from the interpolated
    // normal (uSmooth picks flat vs smooth interpolation).  No
    // specular term: sand is matte, and specular is not what
    // "interpolated normals" was asking for.
    // Distance haze toward the sky colour, capped at 90%: a
    // pure function of true distance, identical in both depth
    // passes, so the near/far boundary stays invisible.
    float dist = length(vWorld - uCamPos);
    col = mix(col, uHazeColor, 0.90 * (1.0 - exp(-dist / 3500.0)));
    fragColor = vec4(col, 1.0);
}
"""

HORIZON_VERT = """
#version 330 core
layout(location = 0) in vec4 aHomog;
layout(location = 1) in vec4 aColor;
uniform mat4 uVP;
uniform vec3 uSunDir;
uniform vec3 uCamPos;
uniform vec3 uHazeColor;
uniform float uDebug;
out vec4 vColor;
void main() {
    vec3 lit = aColor.rgb;
    float hazeF = 0.90;
    if (uDebug > 0.5) {
        hazeF = 0.0;
    } else if (aHomog.w > 0.5) {
        float dist = length(aHomog.xyz - uCamPos);
        hazeF = 0.90 * (1.0 - exp(-dist / 3500.0));
    }
    if (uDebug < 0.5) {
        // Representative-normal lighting for a horizon point:
        // integrate the diffuse response over normals tilted
        // from straight up to 45 degrees toward the viewer.
        // The far terrain actually presents such a fan of
        // normals -- every ridge facing us at every tilt, with
        // lower ground occluded behind it -- so the horizon
        // colour is that fan's average response, not the
        // flat-up response (which reads too bright).
        vec2 toViewer = aHomog.w > 0.5
            ? normalize(uCamPos.xz - aHomog.xz)
            : normalize(-aHomog.xz);
        vec3 sunDir = normalize(uSunDir);
        vec3 acc = vec3(0.0);
        for (int i = 0; i < 4; i++) {
            float tilt = radians(15.0 * float(i));
            vec3 n = normalize(vec3(toViewer.x * sin(tilt), cos(tilt),
                                    toViewer.y * sin(tilt)));
            vec3 hemi = mix(vec3(0.36, 0.32, 0.28), vec3(0.50, 0.64, 0.88),
                            0.5 + 0.5 * n.y);
            acc += hemi + vec3(0.16)
                 + vec3(1.0, 0.93, 0.80) * max(dot(n, sunDir), 0.0) * 0.95;
        }
        lit = aColor.rgb * (acc / 4.0);
        lit = mix(lit, uHazeColor, hazeF);
    }
    vColor = vec4(lit, aColor.a);
    gl_Position = uVP * aHomog;
}
"""

HORIZON_FRAG = """
#version 330 core
in vec4 vColor;
out vec4 fragColor;
void main() {
    fragColor = vColor;
}
"""
