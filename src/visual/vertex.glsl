#version 450 core

// Uniform attribute
uniform int uTotalLines;

// Input attributes
layout(location = 0) in int  aIndex;
layout(location = 1) in vec2 aPosition;

// Vertex color
flat out vec4 vColor;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main( ) {
    gl_Position = vec4(aPosition, 0.0f, 1.0f);

    float h = float(aIndex) / float(max(1, uTotalLines));
    float s = 0.75 + 0.25 * sin(h * 6.2831);  // ~0.5 to 1.0
    float v = 0.85 + 0.15 * cos(h * 3.1416);   // ~0.7 to 1.0
    vec3 rgb_color = hsv2rgb(vec3(h, s, v));

    vColor = vec4(rgb_color, 1.0f);
}