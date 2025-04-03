#version 450 core

// SSBO attribute
layout(std430, binding=0) buffer ssbo {
    const vec3 colors[];
};

// Input attributes
layout(location = 0) in int  aIndex;
layout(location = 1) in vec2 aPosition;

out vec4 vColor;

void main( ) {
    gl_Position = vec4(aPosition, 0.0f, 1.0f);
    
    vColor = vec4(colors[aIndex], 1.0f);
}