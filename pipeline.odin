// Pipeline (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"

BlendMode :: enum u32 {
	NONE                = 0,
	BLEND               = 1,
	BLEND_PREMULTIPLIED = 16,
	ADD                 = 2,
	ADD_PREMULTIPLIED   = 32,
	MOD                 = 4,
	MUL                 = 8,
	SIZE                = 7,
}

PrimitiveType :: enum u32 {
	TRIANGLES      = 0,
	TRIANGLE_STRIP = 1,
	LINES          = 2,
	LINE_STRIP     = 3,
	POINTS         = 4,
	SIZE           = 5,
}

Pipeline :: struct {
	id: u32,
}

// Create a graphics pipeline, Returns an invalid pipeline if creation failed,
// Use GetLastError() to get more information about the error.
CreatePipeline :: proc(shader_vert, shader_frag: Shader, primitive_type: PrimitiveType, blend_mode: BlendMode) -> Pipeline {
	assert(_pipeline_ctx.initialized == _INIT_COOKIE)

	// Location 0 packs position.xy + texcoord as one FLOAT4 ("coord" in
	// shaders/painter.vert.glsl); location 1 is the normalized color.
	vertex_buffer_descriptions := [1]sdl.GPUVertexBufferDescription{{
		slot               = 0,
		pitch              = u32(size_of(Vertex)),
		input_rate         = .VERTEX,
		instance_step_rate = 0,
	}}

	vertex_attributes := [2]sdl.GPUVertexAttribute{
		{location = 0, buffer_slot = 0, format = .FLOAT4, offset = u32(offset_of(Vertex, position))},
		{location = 1, buffer_slot = 0, format = .UBYTE4_NORM, offset = u32(offset_of(Vertex, color))},
	}

	vertex_input_state := sdl.GPUVertexInputState{
		vertex_buffer_descriptions = &vertex_buffer_descriptions[0],
		num_vertex_buffers         = 1,
		vertex_attributes          = &vertex_attributes[0],
		num_vertex_attributes      = 2,
	}

	blend_state := _pipeline_blend_state(blend_mode)

	color_target_descriptions := [1]sdl.GPUColorTargetDescription{{
		format      = sdl.GetGPUSwapchainTextureFormat(_pipeline_ctx.gpu_device, _pipeline_ctx.window),
		blend_state = blend_state,
	}}

	target_info := sdl.GPUGraphicsPipelineTargetInfo{
		color_target_descriptions = &color_target_descriptions[0],
		num_color_targets         = 1,
	}

	pipeline_create_info := sdl.GPUGraphicsPipelineCreateInfo{
		vertex_shader      = GetGPUShader(shader_vert),
		fragment_shader    = GetGPUShader(shader_frag),
		vertex_input_state = vertex_input_state,
		primitive_type     = sdl.GPUPrimitiveType(primitive_type),
		target_info        = target_info,
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(_pipeline_ctx.gpu_device, pipeline_create_info)

	if pipeline == nil {
		_set_error(.CREATE_PIPELINE_FAILED)
		return Pipeline{id = INVALID_ID}
	}

	slot := AcquirePoolSlot(_pipeline_ctx.pool)
	if slot == POOL_INVALID_SLOT {
		_set_error(.CREATE_PIPELINE_FAILED)
		return Pipeline{id = INVALID_ID}
	}

	_pipeline_ctx.pipelines[slot] = _Pipeline{
		pipeline = pipeline,
	}

	return Pipeline{id = GeneratePoolId(_pipeline_ctx.pool, slot)}
}

// Destroy a graphics pipeline and free its resources.
DestroyPipeline :: proc(pipeline: Pipeline) {
	assert(_pipeline_ctx.initialized == _INIT_COOKIE)

	if pipeline.id == INVALID_ID {
		return
	}

	slot := PoolIdToSlot(pipeline.id)

	inner_pipeline := _pipeline_ctx.pipelines[slot].pipeline
	sdl.ReleaseGPUGraphicsPipeline(_pipeline_ctx.gpu_device, inner_pipeline)

	ReleasePoolSlot(_pipeline_ctx.pool, slot)

	_pipeline_ctx.pipelines[slot] = _Pipeline{
		pipeline = nil,
	}
}

// Get the GPU graphics pipeline associated with a SDL_gp pipeline. Returns
// NULL if the pipeline is invalid.
GetGPUPipeline :: proc(pipeline: Pipeline) -> ^sdl.GPUGraphicsPipeline {
	assert(_pipeline_ctx.initialized == _INIT_COOKIE)

	if pipeline.id == INVALID_ID {
		return nil
	}

	slot := PoolIdToSlot(pipeline.id)
	return _pipeline_ctx.pipelines[slot].pipeline
}

// Pipeline (Private)
// ----------------------------------------------------------------------------

@(private)
_Pipeline :: struct {
	pipeline: ^sdl.GPUGraphicsPipeline,
}

@(private)
_PipelineContext :: struct {
	initialized: u32,
	pipelines:   []_Pipeline,
	pool:        ^Pool,
	gpu_device:  ^sdl.GPUDevice,
	window:      ^sdl.Window,
}

@(private)
_pipeline_ctx: _PipelineContext

// Setup pipeline resources management.
@(private)
_PipelineSetup :: proc(gpu_device: ^sdl.GPUDevice, window: ^sdl.Window) {
	assert(_pipeline_ctx.initialized == 0)
	assert(gpu_device != nil)
	assert(window != nil)

	_pipeline_ctx.initialized = _INIT_COOKIE
	_pipeline_ctx.gpu_device = gpu_device
	_pipeline_ctx.window = window

	_pipeline_ctx.pool = CreatePool(PIPELINE_MAX)
	_pipeline_ctx.pipelines = make([]_Pipeline, PIPELINE_MAX)
}

// Shutdown pipeline resources management and free resources.
@(private)
_PipelineShutdown :: proc() {
	assert(_pipeline_ctx.initialized == _INIT_COOKIE)
	_pipeline_ctx.initialized = 0

	DestroyPool(_pipeline_ctx.pool)
	delete(_pipeline_ctx.pipelines)
}

@(private)
_pipeline_blend_state :: proc(blend_mode: BlendMode) -> sdl.GPUColorTargetBlendState {
	blend := sdl.GPUColorTargetBlendState{}

	switch blend_mode {
	case .BLEND:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .SRC_ALPHA
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ONE
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .BLEND_PREMULTIPLIED:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .ONE
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ONE
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .ADD:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .SRC_ALPHA
		blend.dst_color_blendfactor = .ONE
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .ADD_PREMULTIPLIED:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .ONE
		blend.dst_color_blendfactor = .ONE
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .MOD:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .DST_COLOR
		blend.dst_color_blendfactor = .ZERO
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .MUL:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .DST_COLOR
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .DST_ALPHA
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .NONE, .SIZE: // default in C covers NONE and any other value
		blend.enable_blend          = false
		blend.src_color_blendfactor = .ONE
		blend.dst_color_blendfactor = .ZERO
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ONE
		blend.dst_alpha_blendfactor = .ZERO
		blend.alpha_blend_op        = .ADD
	}

	return blend
}
