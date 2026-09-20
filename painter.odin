// Painter (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"
import "core:c"
import "core:math"
import "core:math/linalg"
import "core:mem"

Uniform_Slot :: enum u32 {VS, FS}

Vertex :: struct {
	position: Vec2,
	texcoord: Vec2,
	color:    Color,
}

#assert(size_of(Vertex) == 20, "Vertex layout changed, update pipeline vertex description")

Textured_Rect :: struct {dst, src: Rect}

Uniform_Data :: union {
	[UNIFORM_FLOATS_MAX]f32,
	[UNIFORM_FLOATS_MAX * size_of(f32)]u8,
}

Uniform :: struct {
	vs_size: u16,
	fs_size: u16,
	data:    Uniform_Data,
}

Texture_Uniform :: struct {
	count:    u32,
	images:   [TEXTURE_SLOTS_MAX]Image,
	samplers: [TEXTURE_SLOTS_MAX]^sdl.GPUSampler,
}

State :: struct {
	projection:   Mat,
	transform:    Mat,
	mvp:          Mat,
	texture:      Texture_Uniform,
	uniform:      Uniform,
	pipeline:     Pipeline,
	blend_mode:   Blend_Mode,
	frame_size:   Vec2i,
	viewport:     Recti,
	scissor:      Recti,
	color:        Color,
	thickness:    f32,
	base_uniform: int,
	base_vertex:  int,
	base_command: int,
}

Desc :: struct {
	max_vertices: u32,
	max_commands: u32,
	window:       ^sdl.Window,
	gpu_device:   ^sdl.GPUDevice,
}

// Painter (Private)
// ----------------------------------------------------------------------------

_Command_Type :: enum u32 {
	None,
	Draw,
	Viewport,
	Scissor,
}

_Draw_Args :: struct {
	region:         Region,
	pipeline:       Pipeline,
	texture:        Texture_Uniform,
	uniform_index:  u32,
	vertex_index:   u32,
	vertices_count: u32,
}

_Command_Args :: struct {
	draw:     _Draw_Args,
	viewport: Recti,
	scissor:  Recti,
}

_Command :: struct {
	cmd:  _Command_Type,
	args: _Command_Args,
}

_Gp :: struct {
	initialized:            bool,
	desc:                   Desc,
	vertex_transfer_buffer: ^sdl.GPUTransferBuffer,
	vertex_data_buffer:     ^sdl.GPUBuffer,
	shader_vert:            Shader,
	shader_frag:            Shader,
	pipelines:              [len(Primitive_Type) * len(Blend_Mode)]Pipeline,
	nearest_samplers:       ^sdl.GPUSampler,
	white_image:            Image,

	// States stack
	states: [dynamic; STATE_MAX]State,
	state:  State,

	// Transforms stack
	transforms: [dynamic; TRANSFORMS_MAX]Mat,

	// configurable in Desc — never-grow dynamic buffers, preallocated at setup.
	// Used == len(), capacity == cap(); frame rollback == resize shrink.
	vertices: [dynamic]Vertex,

	// configurable in Desc
	commands: [dynamic]_Command,

	// Desc Uniforms stack
	uniforms: [dynamic]Uniform,
}

_gp: _Gp

@(private)
_pipeline_index :: proc (primitive_type: Primitive_Type, blend_mode: Blend_Mode) -> int {
	return int(primitive_type) * len(Blend_Mode) + int(blend_mode)
}

@(private)
_find_or_create_pipeline :: proc (primitive_type: Primitive_Type, blend_mode: Blend_Mode) -> Pipeline {

	index := _pipeline_index(primitive_type, blend_mode)
	pipeline := _gp.pipelines[index]

	if pipeline == {} {
		pipeline = make_pipeline(_gp.shader_vert, _gp.shader_frag, primitive_type, blend_mode)
		_gp.pipelines[index] = pipeline
	}

	return pipeline
}

// Setup painter context. Returns false if setup failed, use get_last_error()
// to get more information about the error.
setup :: proc (desc: ^Desc, allocator := context.allocator) -> bool {
	assert(!_gp.initialized)
	assert(desc != nil)

	_last_error = .None

	_gp.initialized = true

	_gp.desc.max_vertices = VERTICES_MAX if desc.max_vertices == 0 else desc.max_vertices
	_gp.desc.max_commands = COMMANDS_MAX if desc.max_commands == 0 else desc.max_commands
	_gp.desc.window       = desc.window
	_gp.desc.gpu_device   = desc.gpu_device

	_gp.vertices = make([dynamic]Vertex,   0, int(_gp.desc.max_vertices), allocator)
	_gp.commands = make([dynamic]_Command, 0, int(_gp.desc.max_commands), allocator)
	_gp.uniforms = make([dynamic]Uniform,  0, int(_gp.desc.max_commands), allocator)

	// Setup resources management for shaders, pipelines and images

	_shader_setup(_gp.desc.gpu_device)
	_pipeline_setup(_gp.desc.gpu_device, _gp.desc.window)
	if !_image_setup(_gp.desc.gpu_device, _gp.desc.window) {
		shutdown()
		return false
	}

	// Create a white texture

	texture_format := sdl.GetGPUSwapchainTextureFormat(_img_ctx.gpu_device, _img_ctx.window)
	pixel_format   := sdl.GetPixelFormatFromGPUTextureFormat(texture_format)
	format_details := sdl.GetPixelFormatDetails(pixel_format)

	white := sdl.MapRGBA(format_details, nil, 255, 255, 255, 255)
	white_pixels := [4]u32{white, white, white, white}

	white_surface := sdl.CreateSurfaceFrom(2, 2, pixel_format, raw_data(white_pixels[:]), c.int(format_details.bytes_per_pixel) * 2)
	if white_surface == nil {
		shutdown()
		_set_error(.Create_White_Texture_Failed)
		return false
	}
	defer sdl.DestroySurface(white_surface)

	_gp.white_image = make_image(white_surface)
	if _gp.white_image == {} {
		shutdown()
		return false
	}

	// Create a GPU transfer buffer for vertex data

	vertex_transfer_buffer_create_info := sdl.GPUTransferBufferCreateInfo{
		usage = .UPLOAD,
		size  = u32(_gp.desc.max_vertices) * u32(size_of(Vertex)),
	}

	_gp.vertex_transfer_buffer = sdl.CreateGPUTransferBuffer(desc.gpu_device, vertex_transfer_buffer_create_info)
	if _gp.vertex_transfer_buffer == nil {
		_set_error(.Create_Transfer_Buffer_Failed)
		return false
	}

	// Create a GPU buffer for vertex data

	vertex_data_buffer_create_info := sdl.GPUBufferCreateInfo{
		size  = u32(_gp.desc.max_vertices) * u32(size_of(Vertex)),
		usage = {.VERTEX},
	}

	_gp.vertex_data_buffer = sdl.CreateGPUBuffer(desc.gpu_device, vertex_data_buffer_create_info)
	if _gp.vertex_data_buffer == nil {
		sdl.ReleaseGPUTransferBuffer(desc.gpu_device, _gp.vertex_transfer_buffer)
		_set_error(.Create_Vertex_Buffer_Failed)
		return false
	}

	// Create nearest sampler

	nearest_sampler_info := sdl.GPUSamplerCreateInfo{
		min_filter     = .NEAREST,
		mag_filter     = .NEAREST,
		mipmap_mode    = .NEAREST,
		address_mode_u = .CLAMP_TO_EDGE,
		address_mode_v = .CLAMP_TO_EDGE,
		address_mode_w = .CLAMP_TO_EDGE,
	}

	_gp.nearest_samplers = sdl.CreateGPUSampler(desc.gpu_device, nearest_sampler_info)

	// Create common shader

	vert, frag, shaders_ok := _create_common_shaders(desc.gpu_device, allocator)
	if !shaders_ok {
		return false
	}
	_gp.shader_vert = vert
	_gp.shader_frag = frag

	// Create common pipelines
	if _find_or_create_pipeline(.Points,     .None)  != {} &&
	   _find_or_create_pipeline(.Points,     .Blend) != {} &&
	   _find_or_create_pipeline(.Lines,      .None)  != {} &&
	   _find_or_create_pipeline(.Lines,      .Blend) != {} &&
	   _find_or_create_pipeline(.Line_Strip, .None)  != {} &&
	   _find_or_create_pipeline(.Line_Strip, .Blend) != {} &&
	   _find_or_create_pipeline(.Triangles,  .None)  != {} &&
	   _find_or_create_pipeline(.Triangles,  .Blend) != {}
	{
		_set_error(.Create_Common_Pipeline_Failed)
		shutdown()
		return false
	}

	return true
}

// Shutdown painter context.
shutdown :: proc (allocator := context.allocator) {
	if !_gp.initialized {
		return
	}

	// Destroy common pipelines

	for i in 0 ..< len(_gp.pipelines) {
		if _gp.pipelines[i] != {} {
			destroy_pipeline(_gp.pipelines[i])
			_gp.pipelines[i] = {}
		}
	}

	// Destroy common shader

	if _gp.shader_vert != {} {
		destroy_shader(_gp.shader_vert)
	}

	if _gp.shader_frag != {} {
		destroy_shader(_gp.shader_frag)
	}

	// Destroy nearest sampler

	if _gp.nearest_samplers != nil {
		sdl.ReleaseGPUSampler(_gp.desc.gpu_device, _gp.nearest_samplers)
	}

	// Destroy vertex data buffer

	if _gp.vertex_data_buffer != nil {
		sdl.ReleaseGPUBuffer(_gp.desc.gpu_device, _gp.vertex_data_buffer)
	}

	// Destroy vertex transfer buffer

	if _gp.vertex_transfer_buffer != nil {
		sdl.ReleaseGPUTransferBuffer(_gp.desc.gpu_device, _gp.vertex_transfer_buffer)
	}

	// Destroy white texture

	if _gp.white_image != {} {
		destroy_image(_gp.white_image)
	}

	// Shutdown resources management for shaders, pipelines and images
	_image_shutdown()
	_pipeline_shutdown()
	_shader_shutdown()

	delete(_gp.uniforms)
	delete(_gp.commands)
	delete(_gp.vertices)

	_gp = {}
}

// Begin recording draw calls for the current frame. This should be called
// after setting up the painter and acquiring a swapchain texture and command
// buffer for the current frame.
// If return false then an error occurred and the frame should be skipped,
// use get_last_error() to get more information about the error.
begin :: proc (size: Vec2i) -> bool {
	assert(_gp.initialized)

	assert(len(_gp.states) < cap(_gp.states))
	append(&_gp.states, _gp.state)

	w, h := f32(size.x), f32(size.y)

	_gp.state.projection = Mat{2.0 / w, 0.0, -1.0, 0.0, -2.0 / h, 1.0}
	_gp.state.transform  = MAT_IDENTITY
	_gp.state.mvp        = _gp.state.projection

	_gp.state.texture.count = 1
	_gp.state.texture.images[0]   = _gp.white_image
	_gp.state.texture.samplers[0] = _gp.nearest_samplers

	invalid_image: Image
	for i in 1 ..< TEXTURE_SLOTS_MAX {
		_gp.state.texture.images[i]   = invalid_image
		_gp.state.texture.samplers[i] = _gp.nearest_samplers
	}

	_gp.state.uniform = {}

	_gp.state.blend_mode = .None

	_gp.state.frame_size = size
	_gp.state.viewport   = {0, size}
	_gp.state.scissor    = {0, {-1, -1}}
	_gp.state.color      = {255, 255, 255, 255}

	_gp.state.thickness    = max(1.0 / w, 1.0 / h)
	_gp.state.base_vertex  = len(_gp.vertices)
	_gp.state.base_uniform = len(_gp.uniforms)
	_gp.state.base_command = len(_gp.commands)

	return true
}

// Flush the recorded draw calls to the GPU. Returns false if an error
// occurred, use get_last_error() to get more information about the error.
flush :: proc (cmd_buffer: ^sdl.GPUCommandBuffer, texture: ^sdl.GPUTexture) -> bool {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(cmd_buffer != nil)
	assert(texture != nil)

	_image_flush(cmd_buffer)

	base_command := _gp.state.base_command
	base_uniform := _gp.state.base_uniform
	base_vertex  := _gp.state.base_vertex
	end_command  := len(_gp.commands)
	end_vertex   := len(_gp.vertices)

	vertices_count := end_vertex - base_vertex // Number of vertices to draw

	// NOTE: rewind to base lengths happens only after the upload + render
	// loop below completes, so all pointers and indices stay valid for the
	// whole pass. On early return the arrays stay extended (data preserved
	// for diagnosis); the next begin() re-captures base from len().

	// Error, Nothing to draw
	if _last_error != .None do return false

	// Nothing to draw
	if end_command <= base_command do return true

	vertex_data := cast([^]Vertex)sdl.MapGPUTransferBuffer(_gp.desc.gpu_device, _gp.vertex_transfer_buffer, true)
	if vertex_data == nil {
		_set_error(.Flush_Failed)
		return false
	}

	copy(vertex_data[:len(_gp.vertices)-base_vertex], _gp.vertices[base_vertex:])

	sdl.UnmapGPUTransferBuffer(_gp.desc.gpu_device, _gp.vertex_transfer_buffer)

	// Copy pass
	// ------------------------------

	copy_pass := sdl.BeginGPUCopyPass(cmd_buffer)

	vertex_transfer_location := sdl.GPUTransferBufferLocation{
		transfer_buffer = _gp.vertex_transfer_buffer,
		offset          = 0,
	}

	vertex_buffer_region := sdl.GPUBufferRegion{
		buffer = _gp.vertex_data_buffer,
		offset = u32(base_vertex) * u32(size_of(Vertex)),
		size   = u32(vertices_count) * u32(size_of(Vertex)),
	}

	sdl.UploadToGPUBuffer(copy_pass, vertex_transfer_location, vertex_buffer_region, true)

	sdl.EndGPUCopyPass(copy_pass)

	// Render pass
	// ------------------------------

	color_target_info := sdl.GPUColorTargetInfo{
		texture     = texture,
		clear_color = {0, 0, 0, 1},
		load_op     = .DONT_CARE,
		store_op    = .STORE,
		cycle       = false,
	}

	render_pass := sdl.BeginGPURenderPass(cmd_buffer, &color_target_info, 1, nil)

	cur_pipeline_id   := transmute(Pipeline)max(u32)
	cur_uniform_index := max(u32)
	cur_image_ids: [TEXTURE_SLOTS_MAX]Image
	for i in 0 ..< TEXTURE_SLOTS_MAX {
		cur_image_ids[i] = transmute(Image)max(u32)
	}

	// Flush commands
	for cmd in _gp.commands[base_command:end_command] {

		#partial switch cmd.cmd {
		case .Draw:
			if vertices_count == 0 {
				break
			}

			draw := cmd.args.draw

			rebind_uniforms, rebind_texture: bool

			// Check if pipeline needs to be changed
			if draw.pipeline != cur_pipeline_id {
				cur_pipeline_id = draw.pipeline

				// Bind pipeline
				sdl.BindGPUGraphicsPipeline(render_pass, get_gpu_pipeline(draw.pipeline))

				// When pipeline changes we need to rebind uniforms and textures
				rebind_uniforms = true
				rebind_texture = true
			}

			// Check if uniform needs to be changed
			if draw.uniform_index != cur_uniform_index {
				cur_uniform_index = draw.uniform_index
				rebind_uniforms = true
			}

			// Check if texture needs to be changed
			image_bindings: [TEXTURE_SLOTS_MAX]sdl.GPUTextureSamplerBinding

			for j in 0 ..< u32(TEXTURE_SLOTS_MAX) {
				image_id: Image

				if j < draw.texture.count {
					image_id = draw.texture.images[j]
				}

				if image_id != cur_image_ids[j] {
					cur_image_ids[j] = image_id
					rebind_texture = true
				}

				if image_id != {} {
					image_bindings[j] = {
						texture = get_image_gpu_texture(draw.texture.images[j]),
						sampler = draw.texture.samplers[j],
					}
				} else {
					image_bindings[j] = {
						texture = get_image_gpu_texture(_gp.white_image),
						sampler = _gp.nearest_samplers,
					}
				}
			}

			// Rebind textures if needed
			if rebind_texture {
				sdl.BindGPUFragmentSamplers(render_pass, 0, &image_bindings[0], TEXTURE_SLOTS_MAX)
			}

			// Rebind uniforms if needed
			if rebind_uniforms && cur_uniform_index != max(u32) {
				uniform := &_gp.uniforms[draw.uniform_index]

				if uniform.vs_size > 0 {
					sdl.PushGPUVertexUniformData(cmd_buffer, u32(Uniform_Slot.VS), rawptr(&uniform.data), u32(uniform.vs_size))
				}
				if uniform.fs_size > 0 {
					sdl.PushGPUFragmentUniformData(cmd_buffer, u32(Uniform_Slot.FS), rawptr(&uniform.data), u32(uniform.fs_size))
				}
			}

			vertex_buffer_binding := sdl.GPUBufferBinding{
				buffer = _gp.vertex_data_buffer,
				offset = draw.vertex_index * u32(size_of(Vertex)),
			}

			// In every case we need to bind vertex buffers
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)

			sdl.DrawGPUPrimitives(render_pass, draw.vertices_count, 1, 0, 0)
		case .Viewport:
			x, y := **Vec2(cmd.args.viewport.pos)
			w, h := **Vec2(cmd.args.viewport.size)
			sdl.SetGPUViewport(render_pass, {x=x, y=y, w=w, h=h})
		case .Scissor:
			sdl.SetGPUScissor(render_pass, {**cmd.args.scissor.pos, **cmd.args.scissor.size})
		}
	}

	sdl.EndGPURenderPass(render_pass)

	// Rewind frame scratch to base lengths now that upload + render are done.
	resize(&_gp.commands, base_command)
	resize(&_gp.uniforms, base_uniform)
	resize(&_gp.vertices, base_vertex)

	return true
}

// End recording draw calls for the current frame.
end :: proc () {
	assert(_gp.initialized)

	_gp.state = pop(&_gp.states)
}

// Painter (Private): batching internals
// ----------------------------------------------------------------------------

@(private)
_next_uniform :: proc () -> ^Uniform {
	if len(_gp.uniforms) < cap(_gp.uniforms) {
		resize(&_gp.uniforms, len(_gp.uniforms) + 1)
		return &_gp.uniforms[len(_gp.uniforms) - 1]
	}
	_set_error(.Uniforms_Full)
	return nil
}

@(private)
_prev_uniform :: proc () -> ^Uniform {
	if len(_gp.uniforms) > 0 {
		return &_gp.uniforms[len(_gp.uniforms) - 1]
	}
	return nil
}

@(private)
_next_vertices :: proc (count: u32) -> [^]Vertex {
	base := len(_gp.vertices)
	if base + int(count) <= cap(_gp.vertices) {
		resize(&_gp.vertices, base + int(count))
		return cast([^]Vertex)&_gp.vertices[base]
	}
	_set_error(.Vertices_Full)
	return nil
}

@(private)
_next_command :: proc () -> ^_Command {
	if len(_gp.commands) < cap(_gp.commands) {
		resize(&_gp.commands, len(_gp.commands) + 1)
		return &_gp.commands[len(_gp.commands) - 1]
	}
	return nil
}

@(private)
_prev_command :: proc (count: u32) -> ^_Command {
	if len(_gp.commands) - _gp.state.base_command >= int(count) {
		return &_gp.commands[len(_gp.commands) - int(count)]
	} else {
		return nil
	}
}

@(private)
_transform :: proc (m: Mat, dst, src: []Vec2) {
	assert(len(dst) >= len(src))
	for &d, i in dst {
		d = transform_point(m, src[i])
	}
}

@(private)
_region_overlaps :: proc (a, b: Region) -> bool {
	return !(a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y)
}

@(private)
_merge_draw_commands :: proc (
	pipeline:       Pipeline,
	texture:        Texture_Uniform,
	uniform:        ^Uniform,
	region:         Region,
	vertex_index:   u32,
	vertices_count: u32,
) -> bool {
	vertices_count := vertices_count
	prev_cmd: ^_Command = nil
	inter_cmds: [OPTIMIZER_DEPTH]^_Command
	inter_cmd_count := 0

	// Find commands that are mergable
	lookup_depht := u32(OPTIMIZER_DEPTH)
	for depth in 0 ..= lookup_depht {
		cmd := _prev_command(depth + 1)

		if cmd == nil {
			break // Stop on nonexistent command
		}

		if cmd.cmd == .None {
			lookup_depht += 1
			continue // Command was optimized, continue looking
		}

		if cmd.cmd != .Draw {
			break // Stop on scissor or viewport
		}

		// Only command with the same pipeline, texture and uniform can be merged
		texture_bytes := mem.any_to_bytes(texture)
		cmd_texture_bytes := mem.any_to_bytes(cmd.args.draw.texture)
		uniform_match := uniform == nil
		if !uniform_match {
			uniform_match = mem.compare(mem.any_to_bytes(uniform^), mem.any_to_bytes(_gp.uniforms[cmd.args.draw.uniform_index])) == 0
		}
		if cmd.args.draw.pipeline == pipeline &&
		   mem.compare(texture_bytes, cmd_texture_bytes) == 0 &&
		   uniform_match {
			prev_cmd = cmd // Found a command to merge with, stop looking
			break
		} else {
			inter_cmds[inter_cmd_count] = cmd
			inter_cmd_count += 1
		}
	}

	// Can't merge if there is no previous command to merge with
	if prev_cmd == nil {
		return false
	}

	// Allow merging only if there are no overlapping regions in between
	overlaps_next := false
	overlaps_prev := false
	prev_region := prev_cmd.args.draw.region
	for i in 0 ..< inter_cmd_count {
		inter_region := inter_cmds[i].args.draw.region

		if _region_overlaps(region, inter_region) {
			overlaps_next = true
			if overlaps_prev {
				return false // Can't merge if there are overlapping regions in
				// between
			}
		}

		if _region_overlaps(prev_region, inter_region) {
			overlaps_prev = true
			if overlaps_next {
				return false // Can't merge if there are overlapping regions in
				// between
			}
		}
	}

	if !overlaps_next {	// Merge with the previous draw command
		if inter_cmd_count > 0 {
			// Can't merge if we don't have enough space for vertices
			if len(_gp.vertices) + int(vertices_count) > cap(_gp.vertices) {
				return false
			}

			prev_end_vertex := prev_cmd.args.draw.vertex_index + prev_cmd.args.draw.vertices_count
			prev_vertices_count := u32(len(_gp.vertices)) - prev_end_vertex

			// Avoid moving too meny vertices, otherwise it can cause performance
			// regression
			if prev_vertices_count > MOVE_VERTICES_MAX {
				return false
			}

			// Re-organized vertices
			mem.copy(&_gp.vertices[prev_end_vertex + vertices_count], &_gp.vertices[prev_end_vertex], int(prev_vertices_count) * size_of(Vertex))
			mem.copy_non_overlapping(&_gp.vertices[prev_end_vertex], &_gp.vertices[vertex_index + vertices_count], int(vertices_count) * size_of(Vertex))

			// Offset vertices of inter_cmds
			for i in 0 ..< inter_cmd_count {
				inter_cmds[i].args.draw.vertex_index += vertices_count
			}
		}

		// Update draw region and vertices
		prev_region = {linalg.min(prev_region.min, region.min), linalg.max(prev_region.max, region.max)}
		prev_cmd.args.draw.vertices_count += vertices_count
		prev_cmd.args.draw.region = prev_region
	} else {	// Merge with the next draw command
		assert(inter_cmd_count > 0)

		// Append new draw command
		cmd := _next_command()
		if cmd == nil {
			return false
		}

		prev_vertices_count := prev_cmd.args.draw.vertices_count

		// Can't merge if we don't have enough space for vertices
		if len(_gp.vertices) + int(vertices_count) > cap(_gp.vertices) {
			return false
		}

		// Avoid moving too meny vertices, otherwise it can cause performance
		// regression
		if prev_vertices_count > MOVE_VERTICES_MAX {
			return false
		}

		// Pre-reserve the tail we are about to duplicate into BEFORE moving
		// anything, so the copies below cannot alias a reallocation. The extra
		// guard covers growth by prev_vertices_count (the old code would have
		// written out of bounds here; this is strictly safer with identical
		// success-path behavior).
		if len(_gp.vertices) + int(prev_vertices_count) > cap(_gp.vertices) {
			return false
		}
		resize(&_gp.vertices, len(_gp.vertices) + int(prev_vertices_count))

		// Re-organized vertices
		mem.copy(&_gp.vertices[vertex_index + prev_vertices_count], &_gp.vertices[vertex_index], int(vertices_count) * size_of(Vertex))
		mem.copy_non_overlapping(&_gp.vertices[vertex_index], &_gp.vertices[prev_cmd.args.draw.vertex_index], int(prev_vertices_count) * size_of(Vertex))

		// Update draw region and vertices
		prev_region = {linalg.min(prev_region.min, region.min), linalg.max(prev_region.max, region.max)}
		// Tail already reserved before the copies above; nothing left to grow.
		vertices_count += prev_vertices_count

		// Configure the new draw command
		cmd.cmd = .Draw
		cmd.args.draw.pipeline       = pipeline
		cmd.args.draw.texture        = texture
		cmd.args.draw.region         = prev_region
		cmd.args.draw.uniform_index  = prev_cmd.args.draw.uniform_index
		cmd.args.draw.vertex_index   = vertex_index
		cmd.args.draw.vertices_count = vertices_count

		// Force skipping the previous draw command
		prev_cmd.cmd = .None
	}
	return true
}

@(private)
_queue_draw :: proc (pipeline: Pipeline, region: Region, vertex_index: u32, vertices_count: u32, primitive_type: Primitive_Type) {
	pipeline := pipeline
	uniform: ^Uniform = nil
	if _gp.state.pipeline != {} {
		pipeline = _gp.state.pipeline
		uniform = &_gp.state.uniform
	}

	// If the region is completely outside of the viewport, skip the draw call
	if region.min.x > 1.0 || region.min.y > 1.0 || region.max.x < -1.0 || region.max.y < -1.0 {
		resize(&_gp.vertices, int(vertex_index)) // rollback allocated vertices
		return
	}

	// Try to merge with previous draw command
	if primitive_type != .Triangle_Strip &&
	   primitive_type != .Line_Strip &&
	   _merge_draw_commands(pipeline, _gp.state.texture, uniform, region, vertex_index, vertices_count) {
		return
	}

	// Try to reuse previous uniform if possible
	uniform_index := max(u32)
	if uniform != nil {
		prev_uniform := _prev_uniform()

		reuse_uniform := prev_uniform != nil && mem.compare(mem.any_to_bytes(prev_uniform^), mem.any_to_bytes(uniform^)) == 0

		if !reuse_uniform {
			next_uniform := _next_uniform()
			if next_uniform == nil {
				resize(&_gp.vertices, int(vertex_index)) // rollback allocated vertices
				return
			}
			next_uniform^ = _gp.state.uniform
		}

		uniform_index = u32(len(_gp.uniforms)) - 1 // - 1 since _next_uniform
		// already extended the length
	}

	// New draw command
	cmd := _next_command()

	if cmd == nil {
		resize(&_gp.vertices, int(vertex_index)) // rollback allocated vertices
		return
	}

	cmd.cmd = .Draw
	cmd.args.draw.pipeline = pipeline
	cmd.args.draw.texture = _gp.state.texture
	cmd.args.draw.region = region
	cmd.args.draw.uniform_index = uniform_index
	cmd.args.draw.vertex_index = vertex_index
	cmd.args.draw.vertices_count = vertices_count
}

@(private)
_draw_solid :: proc (primitive_type: Primitive_Type, vertices: []Vec2) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	if len(vertices) == 0 do return

	// Setup vertices
	vertex_index := u32(len(_gp.vertices))
	vertices_count := u32(len(vertices))
	v := _next_vertices(vertices_count)
	if v == nil do return

	width := _gp.state.thickness if primitive_type in bit_set[Primitive_Type]{.Points, .Lines, .Line_Strip} else 1.0
	color := _gp.state.color
	mvp   := _gp.state.mvp
	lo    := Vec2(max(f32))
	hi    := Vec2(-max(f32))
	pad   := Vec2(width)

	for pos, i in vertices {
		p := transform_point(mvp, pos)

		lo = linalg.min(lo, p - pad)
		hi = linalg.max(hi, p + pad)

		v[i] = {p, 0, color}
	}

	pipeline := _find_or_create_pipeline(primitive_type, _gp.state.blend_mode)

	// Queue draw
	_queue_draw(pipeline, {lo, hi}, vertex_index, vertices_count, primitive_type)
}

// Get the current transform matrix.
get_matrix :: proc () -> Mat {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	return _gp.state.transform
}

// Set the current transform matrix (recomputes the MVP immediately).
set_matrix :: proc (m: Mat) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	_gp.state.transform = m
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

// Get the current transform as a homogeneous 3x3 matrix (interop boundary).
get_mat3 :: proc () -> Mat3 {
	return to_mat3(get_matrix())
}

// Set the current transform from a homogeneous 3x3 matrix (interop boundary).
set_mat3 :: proc (m: Mat3) {
	set_matrix(from_mat3(m))
}

matrix_set :: set_matrix
matrix_get :: get_matrix
mat3_set   :: set_mat3
mat3_get   :: get_mat3

// Set the coordinate space boundaries in the current viewport.
set_projection :: proc (left, right, bottom, top: f32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	width := right - left
	height := top - bottom

	_gp.state.projection = Mat{2.0 / width, 0.0, -(right + left) / width, 0.0, 2.0 / height, -(top + bottom) / height}

	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

// Reset the projection to the default coordinate space, which is the
// coordinate of the current viewport.
reset_projection :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	w, h := **Vec2(_gp.state.viewport.size)

	_gp.state.projection = Mat{2 / w, 0, -1, 0, -2 / h, 1}

	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

projection_set   :: set_projection
projection_reset :: reset_projection

// Save the current transform matrix on the transform stack. To be pop later
// with pop_transform.
push_transform :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(len(_gp.transforms) < cap(_gp.transforms))

	append(&_gp.transforms, _gp.state.transform)
}

// Restore the transform matrix from the top of the transform stack.
pop_transform :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(len(_gp.transforms) > 0)

	_gp.state.transform = pop(&_gp.transforms)
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

// Scoped transform guard: pushes and auto-pops at scope end, safe on early return.
// Usage: if transform_scope() {...}
@(deferred_none=pop_transform)
transform_scope :: proc () -> bool {
	push_transform()
	return true
}

// Set the current transform matrix to identity (no transformation).
reset_transform :: proc () {
	set_matrix(MAT_IDENTITY)
}

transform_push  :: push_transform
transform_pop   :: pop_transform
transform_reset :: reset_transform

translate_xy :: proc (x, y: f32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	_gp.state.transform[0, 2] += x * _gp.state.transform[0, 0] + y * _gp.state.transform[0, 1]
	_gp.state.transform[1, 2] += x * _gp.state.transform[1, 0] + y * _gp.state.transform[1, 1]
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

translate_vec :: proc (offset: Vec2) {
	translate_xy(offset.x, offset.y)
}

translate :: proc{translate_xy, translate_vec}

rotate :: proc (angle: f32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	c := math.cos(angle)
	s := math.sin(angle)
	t := _gp.state.transform
	_gp.state.transform = Mat{
		c * t[0, 0] + s * t[0, 1], -s * t[0, 0] + c * t[0, 1], t[0, 2],
		c * t[1, 0] + s * t[1, 1], -s * t[1, 0] + c * t[1, 1], t[1, 2],
	}
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

rotate_at :: proc (angle, ax, ay: f32) {
	translate_xy(ax, ay)
	rotate(angle)
	translate_xy(-ax, -ay)
}

scale_xy :: proc (sx, sy: f32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	_gp.state.transform[0, 0] *= sx
	_gp.state.transform[0, 1] *= sy
	_gp.state.transform[1, 0] *= sx
	_gp.state.transform[1, 1] *= sy
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

scale_vec :: proc (s: Vec2) {
	scale_xy(s.x, s.y)
}

scale :: proc{scale_xy, scale_vec}

scale_at :: proc (sx, sy, ax, ay: f32) {
	translate_xy(ax, ay)
	scale_xy(sx, sy)
	translate_xy(-ax, -ay)
}

// Set the current graphics pipeline.
set_pipeline :: proc (pipeline: Pipeline) {
	assert(_gp.initialized)

	_gp.state.pipeline = pipeline

	// Reset uniforms when pipeline changes
	_gp.state.uniform = {}
}

// Reset the graphics pipeline to the default pipeline builtin pipeline.
reset_pipeline :: proc () {
	assert(_gp.initialized)

	set_pipeline({})
}

pipeline_set   :: set_pipeline
pipeline_reset :: reset_pipeline

// Set uniform data for the current pipeline.
set_uniform :: proc (vs_data: rawptr, vs_size: i32, fs_data: rawptr, fs_size: i32) {
	assert(_gp.initialized)
	assert(_gp.state.pipeline != {})

	size := int(vs_size) + int(fs_size)

	assert(size <= UNIFORM_FLOATS_MAX * size_of(f32))

	if vs_size > 0 {
		assert(vs_data != nil)
		mem.copy(&_gp.state.uniform.data, vs_data, int(vs_size))
	}
	if fs_size > 0 {
		assert(fs_data != nil)
		mem.copy(mem.ptr_offset(cast([^]u8)&_gp.state.uniform.data, int(vs_size)), fs_data, int(fs_size))
	}

	old_size := int(_gp.state.uniform.vs_size) + int(_gp.state.uniform.fs_size)

	if old_size > size {
		// Zero out the rest of the uniform data
		mem.zero(mem.ptr_offset(cast([^]u8)&_gp.state.uniform.data, size), old_size - size)
	}

	_gp.state.uniform.vs_size = u16(vs_size)
	_gp.state.uniform.fs_size = u16(fs_size)
}

// Reset uniform data to the default state (current state color).
reset_uniform :: proc () {
	assert(_gp.initialized)
	assert(_gp.state.pipeline != {})

	set_uniform(nil, 0, nil, 0)
}

uniform_set   :: set_uniform
uniform_reset :: reset_uniform

// Set the current blend mode.
set_blend_mode :: proc (blend_mode: Blend_Mode) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	_gp.state.blend_mode = blend_mode
}

// Reset the current blend mode to the default blend mode (no blending).
reset_blend_mode :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	_gp.state.blend_mode = .None
}

blend_mode_set   :: set_blend_mode
blend_mode_reset :: reset_blend_mode

// Sets current color.
set_color :: proc (color: Color) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	_gp.state.color = color
}

// Gets current color.
get_color :: proc () -> Color {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	return _gp.state.color
}

// Reset current color to the default color (white).
reset_color :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	_gp.state.color = 255
}

color_set   :: set_color
color_get   :: get_color
color_reset :: reset_color

// Sets current bound image in a texture channel.
set_image :: proc (channel: i32, image: Image) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	ch := int(channel)
	if _gp.state.texture.images[ch] == image {
		return
	}

	_gp.state.texture.images[ch] = image

	// Recalculate texture count
	texture_count := int(_gp.state.texture.count)
	for i := max(ch, texture_count - 1); i >= 0; i -= 1 {
		if _gp.state.texture.images[i] != {} {
			texture_count = i + 1
			break
		}
	}

	_gp.state.texture.count = u32(texture_count)
}

// Reset current bound image in a texture channel to the default (white
// texture).
reset_image :: proc (channel: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	set_image(channel, _gp.white_image)
}

image_set   :: set_image
image_reset :: reset_image

// Remove current bound image from a texture channel (no texture).
unset_image :: proc (channel: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	set_image(channel, {})
}

// Set current bound sampler in a texture channel.
set_sampler :: proc (channel: i32, sampler: ^sdl.GPUSampler) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[channel] = sampler
}

// Remove current bound sampler from a texture channel (no sampler).
unset_sampler :: proc (channel: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[channel] = nil
}

// Reset current bound sampler in a texture channel to default (nearest
// sampler).
reset_sampler :: proc (channel: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[channel] = _gp.nearest_samplers
}

sampler_set   :: set_sampler
sampler_reset :: reset_sampler

// Set the screen are to draw to.
set_viewport_xy :: proc (x, y, w, h: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	viewport := Recti{{x, y}, {w, h}}

	// If no change in viewport, skip
	if _gp.state.viewport == viewport {
		return
	}

	// Try to reuse previous command
	cmd := _prev_command(1)
	if cmd != nil && cmd.cmd != .Viewport {
		cmd = _next_command()
	}
	if cmd == nil {
		return
	}

	cmd.cmd = .Viewport
	cmd.args.viewport = viewport

	// When viewport changes, scissor needs to be updated to keep the same region
	if !(_gp.state.scissor.size.x < 0 && _gp.state.scissor.size.y < 0) {
		_gp.state.scissor.pos += Vec2i{x, y} - _gp.state.viewport.pos
	}

	_gp.state.viewport = viewport
	_gp.state.thickness = max(1 / f32(w), 1 / f32(h))
	fw := f32(w)
	fh := f32(h)
	_gp.state.projection = Mat{2 / fw, 0, -1, 0, -2 / fh, 1}
	_gp.state.mvp = compose(_gp.state.projection, _gp.state.transform)
}

// Set the screen area to draw to from an integer rect (shares the body above).
set_viewport_rect :: proc (r: Recti) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	set_viewport_xy(r.pos.x, r.pos.y, r.size.x, r.size.y)
}

set_viewport :: proc{set_viewport_xy, set_viewport_rect}

// Reset the viewport to default (0, 0, width, height).
reset_viewport :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	set_viewport(0, 0, _gp.state.frame_size.x, _gp.state.frame_size.y)
}

viewport_set   :: set_viewport
viewport_reset :: reset_viewport

// Set the clipping rectangle in the viewport.
set_scissor_xy :: proc (x, y, w, h: i32) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	scissor := Recti{{x, y}, {w, h}}

	// Skip if scissor is the same
	if _gp.state.scissor == scissor {
		return
	}

	// Try to reuse previous command
	cmd := _prev_command(1)
	if cmd != nil && cmd.cmd != .Scissor {
		cmd = _next_command()
	}
	if cmd == nil {
		return
	}

	// Coordinates scissor relative to viewport
	viewport_scissor := Recti{_gp.state.viewport.pos + Vec2i{x, y}, {w, h}}

	// Reset scissor
	if w < 0 && h < 0 {
		viewport_scissor.pos = {0, 0}
		viewport_scissor.size = {_gp.state.frame_size.x, _gp.state.frame_size.y}
	}

	cmd.cmd = .Scissor
	cmd.args.scissor = viewport_scissor

	_gp.state.scissor = scissor
}

// Set the clipping rectangle from an integer rect (shares the body above).
set_scissor_rect :: proc (r: Recti) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	set_scissor_xy(r.pos.x, r.pos.y, r.size.x, r.size.y)
}

set_scissor :: proc{set_scissor_xy, set_scissor_rect}

// Reset the clipping rectangle to default (viewport bounds).
reset_scissor :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	_gp.state.scissor = Recti{{0, 0}, {-1, -1}}
}

scissor_set   :: set_scissor
scissor_reset :: reset_scissor

// Reset all state to default.
reset_state :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	reset_viewport()
	reset_scissor()
	reset_projection()
	reset_transform()
	reset_blend_mode()
	reset_color()
	reset_uniform()
	reset_pipeline()
}

state_reset :: reset_state

// Clear the current viewport with the current color.
clear :: proc () {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	// Setup vertices
	vertices_count := u32(6)
	vertex_index := u32(len(_gp.vertices))

	v := _next_vertices(vertices_count)
	if v == nil do return

	// Compute vertices
	quad := [4]Vec2{
		{-1.0, -1.0}, // bottom-left
		{ 1.0, -1.0}, // bottom-right
		{ 1.0,  1.0}, // top-right
		{-1.0,  1.0}, // top-left
	}

	texcoord: Vec2
	color := _gp.state.color

	v[0] = {quad[0], texcoord, color}
	v[1] = {quad[1], texcoord, color}
	v[2] = {quad[2], texcoord, color}
	v[3] = {quad[2], texcoord, color}
	v[4] = {quad[3], texcoord, color}
	v[5] = {quad[0], texcoord, color}

	pipeline := _find_or_create_pipeline(.Triangles, _gp.state.blend_mode)

	_queue_draw(pipeline, Region{-1, 1}, vertex_index, vertices_count, .Triangles)
}

// Draw any primitive.
draw :: proc (primitive_type: Primitive_Type, vertices: []Vertex) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	if len(vertices) == 0 {
		return
	}
	vertices_count := u32(len(vertices))

	// Setup vertices
	vertex_index := u32(len(_gp.vertices))
	v := _next_vertices(vertices_count)
	if v == nil do return

	mvp := _gp.state.mvp
	lo  := Vec2(max(f32))
	hi  := Vec2(-max(f32))

	width := _gp.state.thickness if primitive_type in bit_set[Primitive_Type]{.Points, .Lines, .Line_Strip} else 1.0
	pad   := Vec2(width)

	for i in 0 ..< vertices_count {
		p := transform_point(mvp, vertices[i].position)

		lo = linalg.min(lo, p - pad)
		hi = linalg.max(hi, p + pad)

		v[i] = {p, vertices[i].texcoord, vertices[i].color}
	}

	region := Region{lo, hi}

	pipeline := _find_or_create_pipeline(primitive_type, _gp.state.blend_mode)

	// Queue draw
	_queue_draw(pipeline, region, vertex_index, vertices_count, primitive_type)
}

// Draw points in batch.
draw_points :: proc (points: []Point) {
	_draw_solid(.Points, points)
}

// Draw a single point.
draw_point_single :: proc (point: Point) {
	draw_points({point})
}

// Draw lines in batch.
draw_lines :: proc (lines: []Line) {
	_draw_solid(.Lines, (cast([^]Vec2)raw_data(lines))[:len(lines)*2])
}

// Draw a single line.
draw_line_single :: proc (line: Line) {
	draw_lines({line})
}

// Draw a stip of lines.
draw_line_strip :: proc (points: []Vec2) {
	_draw_solid(.Line_Strip, points)
}

// Draw triangles in batch.
draw_triangles :: proc (triangles: []Triangle) {
	_draw_solid(.Triangles, (cast([^]Vec2)raw_data(triangles))[:len(triangles)*3])
}

// Draw a single triangle.
draw_triangle_single :: proc (triangle: Triangle) {
	draw_triangles({triangle})
}

// Draw a strip of triangles.
draw_triangle_strip :: proc (points: []Vec2) {
	_draw_solid(.Triangle_Strip, points)
}

// Draw rectangles in batch.
draw_rects :: proc (rects: []Rect) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)

	if len(rects) == 0 {
		return
	}

	// Setup vertices
	total_vertices := u32(len(rects)) * 6 // 2 triangles per rect, 3 vertices each
	vertex_index := u32(len(_gp.vertices))
	v := _next_vertices(total_vertices)
	if v == nil {
		return
	}

	// Compute vertices
	color := _gp.state.color
	mvp := _gp.state.mvp
	lo := Vec2(max(f32))
	hi := Vec2(-max(f32))

	for i in 0 ..< len(rects) {
		rect := &rects[i]
		quad := [4]Vec2{
			rect.pos + {0, rect.size.y}, // bottom-left
			rect.pos + rect.size, // bottom-right
			rect.pos + {rect.size.x, 0}, // top-right
			rect.pos, // top-left
		}

		_transform(mvp, quad[:], quad[:])

		for q in quad {
			lo = linalg.min(lo, q)
			hi = linalg.max(hi, q)
		}

		texcoords := [4]Vec2{
			{0.0, 1.0}, // bottom-left
			{1.0, 1.0}, // bottom-right
			{1.0, 0.0}, // top-right
			{0.0, 0.0}, // top-left
		}

		// Make two triangles to form the quad
		v[i * 6 + 0] = {quad[0], texcoords[0], color}
		v[i * 6 + 1] = {quad[1], texcoords[1], color}
		v[i * 6 + 2] = {quad[2], texcoords[2], color}
		v[i * 6 + 3] = {quad[3], texcoords[3], color}
		v[i * 6 + 4] = {quad[0], texcoords[0], color}
		v[i * 6 + 5] = {quad[2], texcoords[2], color}
	}

	// Queue draw
	pipeline := _find_or_create_pipeline(.Triangles, _gp.state.blend_mode)

	_queue_draw(pipeline, {lo, hi}, vertex_index, total_vertices, .Triangles)
}

// Draw a single rectangle.
draw_rect_single :: proc (rect: Rect) {
	draw_rects({rect})
}

// Draw a single rectangle from position + size vectors.
draw_rect_vec :: proc (pos, size: Vec2) {
	draw_rect_single({pos, size})
}

// Draw a single rectangle from x, y, width, height.
draw_rect_xywh :: proc (x, y, w, h: f32) {
	draw_rect_single({{x, y}, {w, h}})
}

// Integer variants.

draw_rect_vec_i :: proc (pos, size: Vec2i) {
	draw_rect_single(rect_to_float({pos, size}))
}
draw_rect_xywh_i :: proc (x, y, w, h: i32) {
	draw_rect_single(rect_to_float({{x, y}, {w, h}}))
}

// Textured variants.

draw_textured_rect_vec :: proc (channel: i32, pos, size: Vec2, src: Rect) {
	draw_textured_rect_single(channel, {{pos, size}, src})
}
draw_textured_rect_xywh :: proc (channel: i32, x, y, w, h: f32, src: Rect) {
	draw_textured_rect_single(channel, {{{x, y}, {w, h}}, src})
}
draw_textured_rect_vec_i :: proc (channel: i32, pos, size: Vec2i, src: Rect) {
	draw_textured_rect_single(channel, {rect_to_float({pos, size}), src})
}
draw_textured_rect_xywh_i :: proc (channel: i32, x, y, w, h: i32, src: Rect) {
	draw_textured_rect_single(channel, {rect_to_float({{x, y}, {w, h}}), src})
}

// Draw textured rectangles in batch.
draw_textured_rects :: proc (channel: i32, rects: []Textured_Rect) {
	assert(_gp.initialized)
	assert(len(_gp.states) > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	if len(rects) == 0 {
		return
	}

	// Setup vertices
	total_vertices := u32(len(rects)) * 6 // 2 triangles per rect, 3 vertices each
	vertex_index := u32(len(_gp.vertices))
	vertices := _next_vertices(total_vertices)
	if vertices == nil do return

	// Get image info
	image := _gp.state.texture.images[int(channel)]
	uv_scale := 1.0 / Vec2(get_image_size(image))

	// Compute vertices
	mvp   := _gp.state.mvp
	color := _gp.state.color
	lo := Vec2(max(f32))
	hi := Vec2(-max(f32))

	for rect, i in rects {
		dst := rect.dst
		quad := [4]Vec2{
			dst.pos + {0, dst.size.y}, // bottom left
			dst.pos + dst.size,        // bottom right
			dst.pos + {dst.size.x, 0}, // top right
			dst.pos,                   // top left
		}

		_transform(mvp, quad[:], quad[:])

		for q in quad {
			lo = linalg.min(lo, q)
			hi = linalg.max(hi, q)
		}

		uv0 := rects[i].src.pos * uv_scale
		uv1 := (rects[i].src.pos + rects[i].src.size) * uv_scale

		vtexquad := [4]Vec2{
			{uv0.x, uv1.y}, // bottom-left
			uv1,            // bottom-right
			{uv1.x, uv0.y}, // top-right
			uv0,            // top-left
		}

		vertices[i * 6 + 0] = {quad[0], vtexquad[0], color}
		vertices[i * 6 + 1] = {quad[1], vtexquad[1], color}
		vertices[i * 6 + 2] = {quad[2], vtexquad[2], color}
		vertices[i * 6 + 3] = {quad[3], vtexquad[3], color}
		vertices[i * 6 + 4] = {quad[0], vtexquad[0], color}
		vertices[i * 6 + 5] = {quad[2], vtexquad[2], color}
	}

	region := Region{lo, hi}

	// Queue draw
	pipeline := _find_or_create_pipeline(.Triangles, _gp.state.blend_mode)

	_queue_draw(pipeline, region, vertex_index, total_vertices, .Triangles)
}

// Draw a single textured rectangle.
draw_textured_rect_single :: proc (channel: i32, rect: Textured_Rect) {
	draw_textured_rects(channel, {rect})
}

// Draw a single integer rect (converts once, shares the float queue path).
draw_rect_i :: proc (rect: Recti) {
	draw_rect_single(rect_to_float(rect))
}

draw_rects_i :: proc (rects: []Recti) {
	for rect in rects {
		draw_rect_single(rect_to_float(rect))
	}
}

draw_textured_rect_i :: proc (channel: i32, dst: Recti, src: Rect) {
	draw_textured_rect_single(channel, {rect_to_float(dst), src})
}

draw_textured_rects_i :: proc (channel: i32, dst: []Recti, src: []Rect) {
	assert(len(dst) == len(src))
	for d, i in dst {
		draw_textured_rect_single(channel, {rect_to_float(d), src[i]})
	}
}

draw_point         :: proc {draw_point_single, draw_points}
draw_line          :: proc {draw_line_single, draw_lines, draw_line_strip}
draw_triangle      :: proc {draw_triangle_single, draw_triangles, draw_triangle_strip}
draw_rect          :: proc {draw_rect_single, draw_rects, draw_rect_i, draw_rects_i, draw_rect_vec, draw_rect_xywh, draw_rect_vec_i, draw_rect_xywh_i}
draw_textured_rect :: proc {draw_textured_rect_single, draw_textured_rects, draw_textured_rect_i, draw_textured_rects_i, draw_textured_rect_vec, draw_textured_rect_xywh, draw_textured_rect_vec_i, draw_textured_rect_xywh_i}
