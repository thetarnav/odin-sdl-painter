// Pipeline (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"

Blend_Mode :: enum u32 {
	None                = 0,
	Blend               = 1,
	Blend_Premultiplied = 16,
	Add                 = 2,
	Add_Premultiplied   = 32,
	Mod                 = 4,
	Mul                 = 8,
}
#assert(len(Blend_Mode) == 7)

Primitive_Type :: enum u32 {
	Triangles,
	Triangle_Strip,
	Lines,
	Line_Strip,
	Points,
}
#assert(len(Primitive_Type) == 5)

Pipeline :: struct {id: u32}

// Create a graphics pipeline, Returns an invalid pipeline if creation failed,
// Use GetLastError() to get more information about the error.
make_pipeline :: proc (shader_vert, shader_frag: Shader, primitive_type: Primitive_Type, blend_mode: Blend_Mode) -> Pipeline {
	assert(_pipeline_ctx.initialized)

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
		vertex_shader      = get_gpu_shader(shader_vert),
		fragment_shader    = get_gpu_shader(shader_frag),
		vertex_input_state = vertex_input_state,
		primitive_type     = sdl.GPUPrimitiveType(primitive_type),
		target_info        = target_info,
	}

	pipeline := sdl.CreateGPUGraphicsPipeline(_pipeline_ctx.gpu_device, pipeline_create_info)

	if pipeline == nil {
		_set_error(.Create_Pipeline_Failed)
		return Pipeline{INVALID_ID}
	}

	slot := acquire_pool_slot(_pipeline_ctx.pool)
	if slot == POOL_INVALID_SLOT {
		_set_error(.Create_Pipeline_Failed)
		return Pipeline{INVALID_ID}
	}

	_pipeline_ctx.pipelines[slot] = {pipeline}

	return Pipeline{id = generate_pool_id(_pipeline_ctx.pool, slot)}
}

// Destroy a graphics pipeline and free its resources.
destroy_pipeline :: proc (pipeline: Pipeline) {
	assert(_pipeline_ctx.initialized)

	if pipeline.id == INVALID_ID do return

	slot := pool_id_to_slot(pipeline.id)

	inner_pipeline := _pipeline_ctx.pipelines[slot].pipeline
	sdl.ReleaseGPUGraphicsPipeline(_pipeline_ctx.gpu_device, inner_pipeline)

	release_pool_slot(_pipeline_ctx.pool, slot)

	_pipeline_ctx.pipelines[slot] = {}
}

// Get the GPU graphics pipeline associated with a SDL_gp pipeline. Returns
// NULL if the pipeline is invalid.
get_gpu_pipeline :: proc (pipeline: Pipeline) -> ^sdl.GPUGraphicsPipeline {
	assert(_pipeline_ctx.initialized)

	if pipeline.id == INVALID_ID do return nil

	slot := pool_id_to_slot(pipeline.id)
	return _pipeline_ctx.pipelines[slot].pipeline
}

// Pipeline (Private)
// ----------------------------------------------------------------------------

_Pipeline :: struct {
	pipeline: ^sdl.GPUGraphicsPipeline,
}

_Pipeline_Context :: struct {
	initialized: bool,
	pipelines:   []_Pipeline,
	pool:        ^Pool,
	gpu_device:  ^sdl.GPUDevice,
	window:      ^sdl.Window,
}

_pipeline_ctx: _Pipeline_Context

// Setup pipeline resources management.
@(private)
_pipeline_setup :: proc (gpu_device: ^sdl.GPUDevice, window: ^sdl.Window, allocator := context.allocator) {
	assert(!_pipeline_ctx.initialized)
	assert(gpu_device != nil)
	assert(window != nil)

	_pipeline_ctx.initialized = true
	_pipeline_ctx.gpu_device  = gpu_device
	_pipeline_ctx.window      = window

	_pipeline_ctx.pool = new_pool(PIPELINE_MAX, allocator)
	_pipeline_ctx.pipelines = make([]_Pipeline, PIPELINE_MAX, allocator)
}

// Shutdown pipeline resources management and free resources.
@(private)
_pipeline_shutdown :: proc (allocator := context.allocator) {
	assert(_pipeline_ctx.initialized)

	delete_pool(_pipeline_ctx.pool, allocator)
	delete(_pipeline_ctx.pipelines, allocator)

	_pipeline_ctx = {}
}

@(private)
_pipeline_blend_state :: proc (blend_mode: Blend_Mode) -> sdl.GPUColorTargetBlendState {
	blend := sdl.GPUColorTargetBlendState{}

	switch blend_mode {
	case .Blend:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .SRC_ALPHA
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ONE
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .Blend_Premultiplied:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .ONE
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ONE
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .Add:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .SRC_ALPHA
		blend.dst_color_blendfactor = .ONE
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .Add_Premultiplied:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .ONE
		blend.dst_color_blendfactor = .ONE
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .Mod:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .DST_COLOR
		blend.dst_color_blendfactor = .ZERO
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .ZERO
		blend.dst_alpha_blendfactor = .ONE
		blend.alpha_blend_op        = .ADD
	case .Mul:
		blend.enable_blend          = true
		blend.src_color_blendfactor = .DST_COLOR
		blend.dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.color_blend_op        = .ADD
		blend.src_alpha_blendfactor = .DST_ALPHA
		blend.dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA
		blend.alpha_blend_op        = .ADD
	case .None: // default in C covers NONE and any other value
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
