#version 450 core

flat in vec4 vColor;

out vec4 fragColor;

void main( ) {
    fragColor = vColor;
}