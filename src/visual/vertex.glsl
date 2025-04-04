#version 450 core

// Uniform attribute
uniform int uTotalLines;

// Input attributes
layout(location = 0) in int  aIndex;
layout(location = 1) in vec2 aPosition;

// Vertex color
out vec4 vColor;

vec3 index_to_rgb(int ix, int total) {
    int spread_ix = (ix / float(total) * (1 << 24));
    char r = (spread_index >> 16) & 255;
    char g = (spread_index >>  8) & 255;
    char b = (spread_index >>  0) & 255;
    return vec4(r, g, b, 1.0f);
}

void main( ) {
    gl_Position = vec4(aPosition, 0.0f, 1.0f);
    
    vColor = index_to_rgb(aIndex, uTotalLines);
}