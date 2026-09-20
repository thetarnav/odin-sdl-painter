// Shader (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import "base:runtime"
import sdl "vendor:sdl3"
import hm "core:container/handle_map"

Shader :: struct {id: u32}

// Create a shader from vertex and fragment shader descriptions. Returns an
// invalid shader if creation failed, Use GetLastError() to get more
// information about the error.
make_shader :: proc (desc: sdl.GPUShaderCreateInfo) -> Shader {
	assert(_shader_ctx.initialized)

	// Create the shader from the bytecode
	sdl_shader := sdl.CreateGPUShader(_shader_ctx.gpu_device, desc)

	if sdl_shader == nil {
		_set_error(.Create_Shader_Failed)
		return Shader{INVALID_ID}
	}

	handle, ok := hm.add(&_shader_ctx.shaders, _Shader{shader = sdl_shader})
	if !ok {
		sdl.ReleaseGPUShader(_shader_ctx.gpu_device, sdl_shader)
		_set_error(.Create_Shader_Failed)
		return Shader{INVALID_ID}
	}

	return Shader{id = _id_from_handle(handle)}
}

// Get the SDL shader associated with a SDL_gp shader. Returns NULL if the
// shader is invalid.
get_gpu_shader :: proc (shader: Shader) -> ^sdl.GPUShader {
	assert(_shader_ctx.initialized)

	if shader.id == INVALID_ID do return nil

	rec, ok := hm.get(&_shader_ctx.shaders, _handle_from_id(shader.id))
	if !ok do return nil
	return rec.shader
}

// Destroy a shader and free its resources.
destroy_shader :: proc (shader: Shader) {
	assert(_shader_ctx.initialized)

	if shader.id == INVALID_ID do return

	handle := _handle_from_id(shader.id)

	rec, ok := hm.get(&_shader_ctx.shaders, handle)
	if !ok do return // stale or foreign id: safe no-op

	sdl.ReleaseGPUShader(_shader_ctx.gpu_device, rec.shader)
	hm.remove(&_shader_ctx.shaders, handle)
}

// Shader (Private)
// ----------------------------------------------------------------------------

// Precompiled painter shaders, selected at runtime by GPU backend.
// GLSL sources in shaders/*.glsl are reference only — SDL3 CreateGPUShader
// consumes backend bytecode, never GLSL.
_vert_spv  := #load("./shaders/painter.vert.spv",  []byte)
_vert_msl  := #load("./shaders/painter.vert.msl",  []byte)
_vert_dxil := #load("./shaders/painter.vert.dxil", []byte)
_frag_spv  := #load("./shaders/painter.frag.spv",  []byte)
_frag_msl  := #load("./shaders/painter.frag.msl",  []byte)
_frag_dxil := #load("./shaders/painter.frag.dxil", []byte)

_Shader :: struct {
	handle: hm.Handle32,
	shader: ^sdl.GPUShader,
}

_Shader_Context :: struct {
	initialized: bool,
	shaders:     hm.Static_Handle_Map(SHADER_MAX, _Shader, hm.Handle32),
	gpu_device:  ^sdl.GPUDevice,
}

_shader_ctx: _Shader_Context

// Setup shader resources management.
@(private)
_shader_setup :: proc (gpu_device: ^sdl.GPUDevice) {
	assert(!_shader_ctx.initialized)
	assert(gpu_device != nil)

	_shader_ctx.initialized = true
	_shader_ctx.gpu_device  = gpu_device
	_shader_ctx.shaders     = {}
}

// Shutdown shader resources management and free resources.
@(private)
_shader_shutdown :: proc () {
	assert(_shader_ctx.initialized)

	it := hm.iterator_make(&_shader_ctx.shaders)
	for {
		rec, _, ok := hm.iterate(&it)
		if !ok do break
		sdl.ReleaseGPUShader(_shader_ctx.gpu_device, rec.shader)
	}

	_shader_ctx = {}
}

// Create the common painter vertex and fragment shaders, selecting the
// precompiled bytecode matching the GPU backend (SPIRV → MSL → DXIL).
// Called from painter Setup. On failure the painter is shut down.
@(private)
_create_common_shaders :: proc (device: ^sdl.GPUDevice, allocator: runtime.Allocator) -> (vert, frag: Shader, ok: bool) {
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
		shutdown(allocator)
		return {}, {}, false
	}

	vert = make_shader({
		code_size  = uint(len(bytecode_vert)),
		code       = raw_data(bytecode_vert),
		entrypoint = "main",
		stage      = .VERTEX,
		format     = format,
	})

	if vert.id == INVALID_ID {
		shutdown(allocator)
		return {}, {}, false
	}

	frag = make_shader({
		code_size    = uint(len(bytecode_frag)),
		code         = raw_data(bytecode_frag),
		entrypoint   = "main",
		stage        = .FRAGMENT,
		format       = format,
		num_samplers = TEXTURE_SLOTS_MAX,
	})

	if frag.id == INVALID_ID {
		shutdown(allocator)
		return {}, {}, false
	}

	return vert, frag, true
}
