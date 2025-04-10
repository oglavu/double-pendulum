#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "mmf.hpp"
#include "gl.hpp"

struct args_t {
    constants_t consts = {
        1, 1,  // l1, l2
        1, 1,  // m1, m2
        0.025, // h
        9.81,  //g
    };

    char srcFilename[100];
};

void read_args(int argc, char* argv[], args_t& myArgs) {

    // read args
    if (argc < 3) {
        printf("Too few args.\n Usage: main.exe <input_file.bin> <num_inst> [<options>]");
        exit(-1);
    }

    strcpy(myArgs.srcFilename, argv[1]);

    myArgs.consts.N = atol(argv[2]);
    
    for (int i=3; i<argc; i += 2) {
        if (argv[i][0] != '-') {
            printf("Bad cmd line args"); exit(-1);
        } 
        if (strcmp(argv[i], "-l1") == 0) {
            myArgs.consts.l1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-l2") == 0) {
            myArgs.consts.l2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m1") == 0) {
            myArgs.consts.m1 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-m2") == 0) {
            myArgs.consts.m2 = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-g") == 0) {
            myArgs.consts.g = atof(argv[i+1]);
        } else if (strcmp(argv[i], "-h") == 0) {
            myArgs.consts.h = atof(argv[i+1]);
        }

    }

}

static void error_callback(int error, const char* description) {
    fputs(description, stderr);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}

static const int WIDTH=640, HEIGHT=480;

int main(int argc, char* argv[]) {
    // TODO: Set M
    args_t myArgs;
    read_args(argc, argv, myArgs);
    myArgs.consts.M = 1<<10;

    GLFWwindow* window;
    if (!glfwInit())
        exit(EXIT_FAILURE);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Double Pendulum Simulator Visual", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        exit(-1);
    }

    physics_kernel::set_constants(myArgs.consts);

    glfwSetErrorCallback(error_callback);
    glfwSetKeyCallback(window, key_callback);

    Shader shader("src/visual/vertex.glsl", "src/visual/fragment.glsl");
    shader.use();
    shader.uniform("uTotalLines", myArgs.consts.N);

    size_t basket_count = myArgs.consts.N;
    size_t basket_size  = basket_count * sizeof(real4_t);

    mmf::mmap_t initMap; initMap.sz = basket_size;
    if (mmf::map_file_rd(myArgs.srcFilename, &initMap) < 0) {
        return -2;
    }

    StripBufferGL buffer(myArgs.consts.N, initMap.h_array);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Black background
    glViewport(0, 0, WIDTH, HEIGHT);
    double currentTime, deltaTime, previousTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {

        currentTime = glfwGetTime();
        deltaTime = currentTime - previousTime;
        if (deltaTime > myArgs.consts.h) {
            previousTime = currentTime;
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            buffer.update();

            buffer.draw();
    
            glfwSwapBuffers(window);
            glfwPollEvents();
        }

    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
