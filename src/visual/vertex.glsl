#version 450 core

// Uniform attribute
uniform int uTotalLines;

// Input attributes
layout(location = 0) in int  aIndex;
layout(location = 1) in vec2 aPosition;

// Vertex color
out vec4 vColor;

void main( ) {
    gl_Position = vec4(aPosition, 0.0f, 1.0f);

    vColor = vec4(1.0, 0.0, 0.0, 1.0);
}