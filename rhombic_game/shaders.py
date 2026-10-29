"""GLSL for the three renderers.  Each renderer has its own math:

- sky:    direction-only.  Positions are directions submitted with
          w = 0; colour is a function of direction alone.
- planes: homogeneous positions.  The fragment shader recovers the
          world point as xyz / w, exact at every distance.
- blocks: the one ordinary finite renderer, for local geometry --
          but still projected with the infinite projection matrix.
"""

SUN_DIR = (-0.45, 0.62, -0.35)  # static daytime sun (normalised in shader)
HAZE_COLOR = (0.74, 0.82, 0.90)

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

_BEDROCK_FRAG = """
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

void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    // 8 m checkerboard, contrast faded as cells shrink below a pixel
    // so it settles to its average instead of shimmering; the pattern
    // is what makes fast flight over the bedrock plane visible.
    vec2 g8 = wp.xz / 8.0;
    vec2 fw = fwidth(g8) + vec2(1e-4);
    float aa = clamp(max(fw.x, fw.y), 0.0, 1.0);
    float check = mod(floor(g8.x) + floor(g8.y), 2.0);
    check = mix(check, 0.5, aa);
    vec3 col = mix(vec3(0.30, 0.28, 0.26), vec3(0.56, 0.53, 0.49), check);

    float line1 = gridLine(wp.xz) * (1.0 - smoothstep(60.0, 500.0, dist));
    float line8 = gridLine(g8) * (1.0 - smoothstep(400.0, 3000.0, dist));
    col = mix(col, vec3(0.16, 0.15, 0.14), line1 * 0.25);
    col = mix(col, vec3(0.10, 0.10, 0.10), line8 * 0.50);

    col = mix(col, uHazeColor, smoothstep(800.0, 7000.0, dist));
    fragColor = vec4(col, 1.0);
}
"""

_SEA_FRAG = """
#version 330 core
in vec4 vHomog;
uniform vec3 uCamPos;
uniform vec3 uSunDir;
uniform vec3 uHazeColor;
out vec4 fragColor;

void main() {
    if (vHomog.w <= 0.0) discard;
    vec3 wp = vHomog.xyz / vHomog.w;
    float dist = length(wp - uCamPos);

    vec2 g = wp.xz / 25.0;
    float check = mod(floor(g.x) + floor(g.y), 2.0);
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

PLANE_FRAGS = {"bedrock": _BEDROCK_FRAG, "sea": _SEA_FRAG}

BLOCK_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uVP;
out vec3 vNormal;
out vec3 vColor;
out vec3 vWorld;
void main() {
    vNormal = aNormal;
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
