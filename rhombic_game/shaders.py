"""GLSL for the three renderers.  Each renderer has its own math:

- sky:    direction-only.  Positions are directions submitted with
          w = 0; colour is a function of direction alone.
- planes: homogeneous positions.  The fragment shader recovers the
          world point as xyz / w, exact at every distance.
- blocks: the one ordinary finite renderer, for local geometry --
          but still projected with the infinite projection matrix.
"""

SUN_DIR = (-0.45, 0.62, -0.35)  # static daytime sun (normalised in shader)
# The sky gradient's horizon colour, also used as the GL clear
# colour.  Nothing in the scene is distance-hazed: haze is a hack
# for masking the limits of finite renderers, and this renderer is
# infinity-first -- the planes run pattern-crisp to the true
# horizon and meet the sky there directly.
HAZE_COLOR = (0.78, 0.86, 0.94)  # (name kept; it is the horizon colour now)

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

    // No distance haze, by design (infinity-first: the pattern
    // runs crisp to the true horizon).
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

    // No distance haze, by design (see the bedrock shader).
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
    // Phong specular: per-pixel view vector against the reflected
    // sun direction.  Sandy surfaces are matte, so the strength
    // is low and the lobe broad -- a sheen on sun-facing facets
    // at grazing view angles, not a wet look.  Gated by ndl so
    // back faces never sparkle.
    vec3 L = normalize(uSunDir);
    vec3 V = normalize(uCamPos - vWorld);
    vec3 R = reflect(-L, n);
    float spec = pow(max(dot(R, V), 0.0), 28.0) * 0.22 * step(0.0, ndl);
    col += vec3(1.0, 0.93, 0.80) * spec;
    fragColor = vec4(col, 1.0);
}
"""

HORIZON_VERT = """
#version 330 core
layout(location = 0) in vec4 aHomog;
layout(location = 1) in vec4 aColor;
uniform mat4 uVP;
out vec4 vColor;
void main() {
    vColor = aColor;
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
