// Shader (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"

Shader :: struct {id: u32}

Shader_Desc :: struct {
	// Vertex shader description
	code_size:            uint,
	code:                 [^]u8,
	entrypoint:           cstring,
	stage:                sdl.GPUShaderStage,
	format:               sdl.GPUShaderFormat,
	num_samplers:         u32,
	num_storage_textures: u32,
	num_storage_buffers:  u32,
	num_uniform_buffers:  u32,
}

// Create a shader from vertex and fragment shader descriptions. Returns an
// invalid shader if creation failed, Use GetLastError() to get more
// information about the error.
create_shader :: proc (desc: ^Shader_Desc) -> Shader {
	assert(_shader_ctx.initialized == _INIT_COOKIE)
	assert(desc != nil)

	create_info := sdl.GPUShaderCreateInfo{
		code_size            = desc.code_size,
		code                 = desc.code,
		entrypoint           = desc.entrypoint,
		format               = desc.format,
		stage                = desc.stage,
		num_samplers         = desc.num_samplers,
		num_storage_textures = desc.num_storage_textures,
		num_storage_buffers  = desc.num_storage_buffers,
		num_uniform_buffers  = desc.num_uniform_buffers,
	}

	// Create the shader from the bytecode
	sdl_shader := sdl.CreateGPUShader(_shader_ctx.gpu_device, create_info)

	if sdl_shader == nil {
		_set_error(.Create_Shader_Failed)
		return Shader{INVALID_ID}
	}

	slot := acquire_pool_slot(_shader_ctx.pool)
	if slot == POOL_INVALID_SLOT {
		sdl.ReleaseGPUShader(_shader_ctx.gpu_device, sdl_shader)
		_set_error(.Create_Shader_Failed)
		return Shader{INVALID_ID}
	}

	_shader_ctx.shader[slot] = _Shader{
		sdl_shader = sdl_shader,
	}

	return Shader{id = generate_pool_id(_shader_ctx.pool, slot)}
}

// Get the SDL shader associated with a SDL_gp shader. Returns NULL if the
// shader is invalid.
get_gpu_shader :: proc (shader: Shader) -> ^sdl.GPUShader {
	assert(_shader_ctx.initialized == _INIT_COOKIE)

	if shader.id == INVALID_ID {
		return nil
	}

	slot := pool_id_to_slot(shader.id)
	return _shader_ctx.shader[slot].sdl_shader
}

// Destroy a shader and free its resources.
destroy_shader :: proc (shader: Shader) {
	assert(_shader_ctx.initialized == _INIT_COOKIE)

	if shader.id == INVALID_ID {
		return
	}

	slot := pool_id_to_slot(shader.id)

	inner_shader := _shader_ctx.shader[slot]

	sdl.ReleaseGPUShader(_shader_ctx.gpu_device, inner_shader.sdl_shader)

	release_pool_slot(_shader_ctx.pool, slot)

	_shader_ctx.shader[slot] = _Shader{
		sdl_shader = nil,
	}
}

// Shader (Private)
// ----------------------------------------------------------------------------

// Precompiled painter shaders, selected at runtime by GPU backend.
// GLSL sources in shaders/*.glsl are reference only — SDL3 CreateGPUShader
// consumes backend bytecode, never GLSL.
_vert_spv  := #load("./shaders/painter.vert.spv", []byte)
_vert_msl  := #load("./shaders/painter.vert.msl", []byte)
_vert_dxil := #load("./shaders/painter.vert.dxil", []byte)
_frag_spv  := #load("./shaders/painter.frag.spv", []byte)
_frag_msl  := #load("./shaders/painter.frag.msl", []byte)
_frag_dxil := #load("./shaders/painter.frag.dxil", []byte)

_INIT_COOKIE :: 0xC0DED1ED

_Shader :: struct {
	sdl_shader: ^sdl.GPUShader,
}

_Shader_Context :: struct {
	initialized: u32,
	shader:      []_Shader,
	pool:        ^Pool,
	gpu_device:  ^sdl.GPUDevice,
}

_shader_ctx: _Shader_Context

// Setup shader resources management.
@(private)
_shader_setup :: proc (gpu_device: ^sdl.GPUDevice, allocator := context.allocator) {
	assert(_shader_ctx.initialized == 0)
	assert(gpu_device != nil)

	_shader_ctx.initialized = _INIT_COOKIE
	_shader_ctx.gpu_device = gpu_device

	_shader_ctx.pool = create_pool(SHADER_MAX, allocator)
	_shader_ctx.shader = make([]_Shader, SHADER_MAX, allocator)
}

// Shutdown shader resources management and free resources.
@(private)
_shader_shutdown :: proc (allocator := context.allocator) {
	assert(_shader_ctx.initialized == _INIT_COOKIE)
	_shader_ctx.initialized = 0

	destroy_pool(_shader_ctx.pool, allocator)
	delete(_shader_ctx.shader, allocator)
}

// Create the common painter vertex and fragment shaders, selecting the
// precompiled bytecode matching the GPU backend (SPIRV → MSL → DXIL).
// Called from painter Setup. On failure the painter is shut down,
// matching the C Setup error paths (SDL_gp.h:2644-2680).
@(private)
_create_common_shaders :: proc (device: ^sdl.GPUDevice) -> (vert, frag: Shader, ok: bool) {
	supported_formats := sdl.GetGPUShaderFormats(device)

	format: sdl.GPUShaderFormat

	bytecode_vert: []byte
	bytecode_frag: []byte

	if .SPIRV in supported_formats {
		format = {.SPIRV}

		bytecode_vert = _vert_spv
		bytecode_frag = _frag_spv
	} else if .MSL in supported_formats {
		format = {.MSL}

		bytecode_vert = _vert_msl
		bytecode_frag = _frag_msl
	} else if .DXIL in supported_formats {
		format = {.DXIL}

		bytecode_vert = _vert_dxil
		bytecode_frag = _frag_dxil
	} else {
		_set_error(.Create_Common_Shader_Failed)
		shutdown()
		return {}, {}, false
	}

	shader_vert_desc := Shader_Desc{
		code_size            = uint(len(bytecode_vert)),
		code                 = raw_data(bytecode_vert),
		entrypoint           = "main",
		stage                = .VERTEX,
		format               = format,
		num_samplers         = 0,
		num_storage_textures = 0,
		num_storage_buffers  = 0,
		num_uniform_buffers  = 0,
	}

	vert = create_shader(&shader_vert_desc)

	if vert.id == INVALID_ID {
		shutdown()
		return {}, {}, false
	}

	shader_frag_desc := Shader_Desc{
		code_size            = uint(len(bytecode_frag)),
		code                 = raw_data(bytecode_frag),
		entrypoint           = "main",
		stage                = .FRAGMENT,
		format               = format,
		num_samplers         = TEXTURE_SLOTS_MAX,
		num_storage_textures = 0,
		num_storage_buffers  = 0,
		num_uniform_buffers  = 0,
	}

	frag = create_shader(&shader_frag_desc)

	if frag.id == INVALID_ID {
		shutdown()
		return {}, {}, false
	}

	return vert, frag, true
}
