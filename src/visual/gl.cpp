#include "gl.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "physics_kernel.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

static const char* glGetErrorString(GLenum error) {
    switch (error) {
    case GL_NO_ERROR:          return "No Error";
    case GL_INVALID_ENUM:      return "Invalid Enum";
    case GL_INVALID_VALUE:     return "Invalid Value";
    case GL_INVALID_OPERATION: return "Invalid Operation";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "Invalid Framebuffer Operation";
    case GL_OUT_OF_MEMORY:     return "Out of Memory";
    case GL_STACK_UNDERFLOW:   return "Stack Underflow";
    case GL_STACK_OVERFLOW:    return "Stack Overflow";
    case GL_CONTEXT_LOST:      return "Context Lost";
    default:                   return "Unknown Error";
    }
}
static void _glCheckErrors(const char *filename, int line) {
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
        printf("OpenGL Error: %s (%d) [%u] %s\n", filename, line, err, glGetErrorString(err));
}
#define glCheckErrors(ans) \
	(ans); _glCheckErrors(__FILE__, __LINE__);


std::string Shader::readFile(std::string filePath) {
	std::ifstream file(filePath);

	if (!file.is_open())
		std::cerr << "Couldnt open VertexShader file " << filePath <<std::endl;

	std::stringstream stream;
	stream << file.rdbuf();

	file.close();

	return stream.str();
}

unsigned Shader::compile(const char* str, ShaderType type) {
	unsigned t = (type == VERTEX) ? GL_VERTEX_SHADER : GL_FRAGMENT_SHADER;
	
	unsigned shaderId = glCreateShader(t);
	glShaderSource(shaderId, 1, &str, 0);
	glCompileShader(shaderId);

	int compilationStatus;
	glGetShaderiv(shaderId, GL_COMPILE_STATUS, &compilationStatus);

	if (compilationStatus != GL_TRUE) {
		int lenght;
		char msg[1024];
		glGetShaderInfoLog(shaderId, 1024, &lenght, msg);
		std::cerr << msg <<std::endl;
	}

	std::cout << "OpenGL: " << ((type == VERTEX) ? "Vertex" : "Fragment" )<< " shader compiled successfully\n";

	return shaderId;
}

unsigned Shader::link(const unsigned vertexId, const unsigned fragmentId) {
	int pLinkingStatus;

	unsigned programId = glCreateProgram();
	glAttachShader(programId, vertexId);
	glAttachShader(programId, fragmentId);
	glLinkProgram(programId);
	glGetProgramiv(programId, GL_LINK_STATUS, &pLinkingStatus);

	if (pLinkingStatus != GL_TRUE) {
		int lenght;
		char msg[1024];
		glGetProgramInfoLog(programId, 1024, &lenght, msg);
		std::cerr << msg <<std::endl;
	}

	std::cout << "OpenGL: Shader linked successfully\n";

	return programId;
}

Shader::Shader(std::string vsFileName, std::string fsFileName) {
	
	// -------- Read ------------
	std::string vsString = Shader::readFile(vsFileName);
	std::string fsString = Shader::readFile(fsFileName);

	const char* vsChar = vsString.c_str();
	const char* fsChar = fsString.c_str();

	// -------- Vertex ------------
	unsigned vertexId = Shader::compile(vsChar, VERTEX);

	// -------- Fragment ------------
	unsigned fragmentId = Shader::compile(fsChar, FRAGMENT);

	// -------- Linking ------------
	m_programId = Shader::link(vertexId, fragmentId);
}

void Shader::use() const {
	glCheckErrors( glUseProgram(m_programId) );
}

void Shader::uniform(const std::string name, int value) const {
	glCheckErrors( glUniform1i(glGetUniformLocation(m_programId, name.data()), value) );
}

StripBufferGL::StripBufferGL(constants_t& consts, void* h_ptr): 
	lines_per_frame(consts.N), 
	frames_per_buffer(consts.M)
{

    glCheckErrors( glGenBuffers(1, &m_vbo[0]) );
	glCheckErrors( glGenBuffers(1, &m_vbo[1]) );
	glCheckErrors( glGenVertexArrays(1, &m_vao) );

	// ix1 x_A1 y_A1 ix1 x_B1 y_B1 ix1 x_C1 y_C1
	// +-------------+ stride1
	//     +-------------+ stride2
	// +---+ offset2

	size_t stride1 = sizeof(vertex_t),
		offset1 = 0,
		stride2 = sizeof(vertex_t),
		offset2 = sizeof(int);

	size_t size = frames_per_buffer * lines_per_frame * 3 * stride1;

	glCheckErrors( glBindVertexArray(m_vao) );
	
	glCheckErrors( glBindBuffer(GL_ARRAY_BUFFER, m_vbo[0]) );
	glCheckErrors( glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW) );
	glCheckErrors( glBindBuffer(GL_ARRAY_BUFFER, m_vbo[1]) );
	glCheckErrors( glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW) );
	
	glCheckErrors( glVertexAttribIPointer(0, 1, GL_INT, stride1, (void*)offset1) );
	glCheckErrors( glEnableVertexAttribArray(0) );
	glCheckErrors( glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride2, (void*)offset2) );
	glCheckErrors( glEnableVertexAttribArray(1) );


	glCheckErrors( glBindVertexArray(0) );
	glCheckErrors( glBindBuffer(GL_ARRAY_BUFFER, 0) );

	gpuErrChk( cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&m_cudaResource[0], m_vbo[0], cudaGraphicsRegisterFlagsWriteDiscard) );
	printf("CUDA: Registered OpenGL buffer (0x%lx)\n", (size_t)m_cudaResource[0]);
	gpuErrChk( cudaGraphicsGLRegisterBuffer((cudaGraphicsResource**)&m_cudaResource[1], m_vbo[1], cudaGraphicsRegisterFlagsWriteDiscard) );
	printf("CUDA: Registered OpenGL buffer (0x%lx)\n", (size_t)m_cudaResource[1]);

	gpuErrChk( cudaMalloc(&m_dInitArray, lines_per_frame * sizeof(real4_t)) );
	gpuErrChk( cudaMemcpy(m_dInitArray, h_ptr, lines_per_frame*sizeof(real4_t), cudaMemcpyHostToDevice) );
	physics_kernel::set_constants(consts);
}

void StripBufferGL::init() {
	cuda_fill(0, false);
	cuda_fill(1);

	m_glIndex = 0;
}

void *StripBufferGL::cuda_map(int ix) {
	if (ix < 0 || ix > 1) return 0;
    
	gpuErrChk( cudaGraphicsMapResources(1, (cudaGraphicsResource**)&m_cudaResource[ix]) );

	size_t size; void* d_ptr;
	gpuErrChk( cudaGraphicsResourceGetMappedPointer(&d_ptr, &size, (cudaGraphicsResource*)m_cudaResource[ix]) );

	printf("CUDA: Buffer {%d} mapped\n", ix);

	return d_ptr;
}

void StripBufferGL::cuda_fill(int ix, bool async) {

	void* d_dataArray = cuda_map(ix);
	m_fillFuture = std::async(
		std::launch::async,
		&physics_kernel::kernel_call,
		1, lines_per_frame, (real4_t*)m_dInitArray, (vertex_t*)d_dataArray
	);

	if (!async) {
		cudaDeviceSynchronize();
		cuda_unmap(ix);
	}

}

void StripBufferGL::cuda_unmap(int ix) {

	// size_t count = 3*lines_per_frame*frames_per_buffer;
	// vertex_t* dst = new vertex_t[count];
	// cudaMemcpy(dst, m_dataArray, count*sizeof(vertex_t), cudaMemcpyDeviceToHost);
	// for (int i=0; i<count; ++i) {
	// 	printf("%d %f %f\n", dst[i].ix, dst[i].x, dst[i].y);
	// }
	// delete dst;

	gpuErrChk( cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t*)&m_cudaResource[ix]) );
	printf("CUDA: Buffer {%d} unmapped\n", ix);
}

void StripBufferGL::update() {
	static const size_t max = 3 * lines_per_frame * frames_per_buffer;
	
	if (m_offset == max) {
		// buffer emptied, time to swap

		int glIndex = (!m_glIndex); // set full buffer for emptying
		int cudaIndex = m_glIndex;  // set empty buffer for refill
		
		if (m_fillFuture.valid()) {
			m_fillFuture.wait(); // wait for a full buffer	
			cuda_unmap(glIndex);
		}

		m_glIndex = glIndex; // switch OpenGL buffer
		m_offset = 0;

		cuda_fill(cudaIndex);

		printf("OpenGL: Switched buffers {%d}->{%d}\n", cudaIndex, glIndex);
	}

}

void StripBufferGL::draw() const {

	static const int count(lines_per_frame);
		
	int sizes[count] {3};
	int starts[count] {(int)m_offset};

	for (int i=1; i<count; ++i) { 
		sizes[i] = 3;
		starts[i] = starts[i-1] + sizes[i];
	}
  
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo[m_glIndex]);

	glMultiDrawArrays(GL_LINE_STRIP, starts, sizes, count);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glCheckErrors(glBindVertexArray(0));

	m_offset += 3*count;
}

StripBufferGL::~StripBufferGL() {
	cudaGraphicsUnregisterResource(*(cudaGraphicsResource_t*)m_cudaResource[0]);
	cudaGraphicsUnregisterResource(*(cudaGraphicsResource_t*)m_cudaResource[1]);
	glDeleteBuffers(1, &m_vbo[0]);
	glDeleteBuffers(1, &m_vbo[1]);
	glDeleteBuffers(1, &m_vao);
}
