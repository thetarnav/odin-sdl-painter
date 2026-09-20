// Pipeline (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"
import hm "core:container/handle_map"

Blend_Mode :: enum u32 {
	None,
	Blend,
	Add,
	Mod,
	Mul,
	Blend_Premultiplied,
	Add_Premultiplied,
}

Primitive_Type :: enum u32 {
	Triangles,
	Triangle_Strip,
	Lines,
	Line_Strip,
	Points,
}

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
		{location = 0, buffer_slot = 0, format = .FLOAT4,      offset = u32(offset_of(Vertex, position))},
		{location = 1, buffer_slot = 0, format = .UBYTE4_NORM, offset = u32(offset_of(Vertex, color))},
	}

	vertex_input_state := sdl.GPUVertexInputState{
		vertex_buffer_descriptions = &vertex_buffer_descriptions[0],
		num_vertex_buffers         = 1,
		vertex_attributes          = &vertex_attributes[0],
		num_vertex_attributes      = 2,
	}

	blend_state := BLEND_STATE[blend_mode]

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

	handle, ok := hm.add(&_pipeline_ctx.pipelines, _Pipeline{pipeline = pipeline})
	if !ok {
		sdl.ReleaseGPUGraphicsPipeline(_pipeline_ctx.gpu_device, pipeline)
		_set_error(.Create_Pipeline_Failed)
		return Pipeline{INVALID_ID}
	}

	return Pipeline{id = _id_from_handle(handle)}
}

// Destroy a graphics pipeline and free its resources.
destroy_pipeline :: proc (pipeline: Pipeline) {
	assert(_pipeline_ctx.initialized)

	if pipeline.id == INVALID_ID do return

	handle := _handle_from_id(pipeline.id)

	rec, ok := hm.get(&_pipeline_ctx.pipelines, handle)
	if !ok do return // stale or foreign id: safe no-op, never aliases a live pipeline

	sdl.ReleaseGPUGraphicsPipeline(_pipeline_ctx.gpu_device, rec.pipeline)
	hm.remove(&_pipeline_ctx.pipelines, handle)
}

// Get the GPU graphics pipeline associated with a SDL_gp pipeline. Returns
// NULL if the pipeline is invalid.
get_gpu_pipeline :: proc (pipeline: Pipeline) -> ^sdl.GPUGraphicsPipeline {
	assert(_pipeline_ctx.initialized)

	if pipeline.id == INVALID_ID do return nil

	rec, ok := hm.get(&_pipeline_ctx.pipelines, _handle_from_id(pipeline.id))
	if !ok do return nil
	return rec.pipeline
}

// Pipeline (Private)
// ----------------------------------------------------------------------------

_Pipeline :: struct {
	handle:   hm.Handle32,
	pipeline: ^sdl.GPUGraphicsPipeline,
}

_Pipeline_Context :: struct {
	initialized: bool,
	pipelines:   hm.Static_Handle_Map(PIPELINE_MAX, _Pipeline, hm.Handle32),
	gpu_device:  ^sdl.GPUDevice,
	window:      ^sdl.Window,
}

_pipeline_ctx: _Pipeline_Context

// Setup pipeline resources management.
@(private)
_pipeline_setup :: proc (gpu_device: ^sdl.GPUDevice, window: ^sdl.Window) {
	assert(!_pipeline_ctx.initialized)
	assert(gpu_device != nil)
	assert(window != nil)

	_pipeline_ctx.initialized = true
	_pipeline_ctx.gpu_device  = gpu_device
	_pipeline_ctx.window      = window
	_pipeline_ctx.pipelines   = {}
}

// Shutdown pipeline resources management and free resources.
@(private)
_pipeline_shutdown :: proc () {
	assert(_pipeline_ctx.initialized)

	it := hm.iterator_make(&_pipeline_ctx.pipelines)
	for {
		rec, _, ok := hm.iterate(&it)
		if !ok do break
		sdl.ReleaseGPUGraphicsPipeline(_pipeline_ctx.gpu_device, rec.pipeline)
	}

	_pipeline_ctx = {}
}

@(private, rodata)
BLEND_STATE := [Blend_Mode]sdl.GPUColorTargetBlendState{
	.None = {},
	.Blend = {
		enable_blend          = true,
		src_color_blendfactor = .SRC_ALPHA,
		dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .ONE,
		dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
		alpha_blend_op        = .ADD,
	},
	.Blend_Premultiplied = {
		enable_blend          = true,
		src_color_blendfactor = .ONE,
		dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .ONE,
		dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
		alpha_blend_op        = .ADD,
	},
	.Add = {
		enable_blend          = true,
		src_color_blendfactor = .SRC_ALPHA,
		dst_color_blendfactor = .ONE,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .ZERO,
		dst_alpha_blendfactor = .ONE,
		alpha_blend_op        = .ADD,
	},
	.Add_Premultiplied = {
		enable_blend          = true,
		src_color_blendfactor = .ONE,
		dst_color_blendfactor = .ONE,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .ZERO,
		dst_alpha_blendfactor = .ONE,
		alpha_blend_op        = .ADD,
	},
	.Mod = {
		enable_blend          = true,
		src_color_blendfactor = .DST_COLOR,
		dst_color_blendfactor = .ZERO,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .ZERO,
		dst_alpha_blendfactor = .ONE,
		alpha_blend_op        = .ADD,
	},
	.Mul = {
		enable_blend          = true,
		src_color_blendfactor = .DST_COLOR,
		dst_color_blendfactor = .ONE_MINUS_SRC_ALPHA,
		color_blend_op        = .ADD,
		src_alpha_blendfactor = .DST_ALPHA,
		dst_alpha_blendfactor = .ONE_MINUS_SRC_ALPHA,
		alpha_blend_op        = .ADD,
	},
}
