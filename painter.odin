// Painter (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"
import "core:c"
import "core:math"
import "core:mem"

UniformSlot :: enum u32 {
	VS = 0,
	FS = 1,
}

Vec2 :: [2]f32

Point :: Vec2

Line :: struct {
	a, b: Point,
}

Triangle :: struct {
	a, b, c: Point,
}

ISize :: struct {
	w, h: i32,
}

IRect :: struct {
	x, y, w, h: i32,
}

Rect :: struct {
	x, y, w, h: f32,
}

TexturedRect :: struct {
	dst, src: Rect,
}

Mat2x3 :: struct {
	m00, m01, m02: f32,
	m10, m11, m12: f32,
}

// Create an immutable 2x3 matrix
create_mat2x3 :: proc(m00, m01, m02, m10, m11, m12: f32) -> Mat2x3 {
	return Mat2x3{m00, m01, m02, m10, m11, m12}
}

// Create an immutable identity 2x3 matrix
create_mat2x3_identity :: proc() -> Mat2x3 {
	return Mat2x3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}
}

Vertex :: struct {
	position: Vec2,
	texcoord: Vec2,
	color:    sdl.Color,
}

when size_of(Vertex) != 20 {
	#panic("SDL_GPVertex layout changed, update pipeline vertex description")
}

UniformData :: union {
	[UNIFORM_FLOATS_MAX]f32,
	[UNIFORM_FLOATS_MAX * size_of(f32)]u8,
}

Uniform :: struct {
	vs_size: u16,
	fs_size: u16,
	data:    UniformData,
}

TextureUniform :: struct {
	count:    u32,
	images:   [TEXTURE_SLOTS_MAX]Image,
	samplers: [TEXTURE_SLOTS_MAX]^sdl.GPUSampler,
}

State :: struct {
	projection:   Mat2x3,
	transform:    Mat2x3,
	mvp:          Mat2x3,
	texture:      TextureUniform,
	uniform:      Uniform,
	pipeline:     Pipeline,
	blend_mode:   BlendMode,
	frame_size:   ISize,
	viewport:     IRect,
	scissor:      IRect,
	color:        sdl.Color,
	thickness:    f32,
	base_uniform: u32,
	base_vertex:  u32,
	base_command: u32,
}

Desc :: struct {
	max_vertices: u32,
	max_commands: u32,
	window:       ^sdl.Window,
	gpu_device:   ^sdl.GPUDevice,
}

// Painter (Private)
// ----------------------------------------------------------------------------

@(private)
_VERTICES_MAX :: 65536
@(private)
_COMMANDS_MAX :: 16384

@(private)
_Region :: struct {
	x1, y1, x2, y2: f32,
}

@(private)
_CommandType :: enum u32 {
	NONE     = 0,
	DRAW     = 1,
	VIEWPORT = 2,
	SCISSOR  = 3,
}

@(private)
_DrawArgs :: struct {
	region:         _Region,
	pipeline:       Pipeline,
	texture:        TextureUniform,
	uniform_index:  u32,
	vertex_index:   u32,
	vertices_count: u32,
}

@(private)
_CommandArgs :: struct {
	draw:     _DrawArgs,
	viewport: IRect,
	scissor:  IRect,
}

@(private)
_Command :: struct {
	cmd:  _CommandType,
	args: _CommandArgs,
}

@(private)
_Gp :: struct {
	initialized:            u32,
	desc:                   Desc,
	vertex_transfer_buffer: ^sdl.GPUTransferBuffer,
	vertex_data_buffer:     ^sdl.GPUBuffer,
	shader_vert:            Shader,
	shader_frag:            Shader,
	pipelines:              [int(PrimitiveType.SIZE) * int(BlendMode.SIZE)]Pipeline,
	nearest_samplers:       ^sdl.GPUSampler,
	white_image:            Image,

	// States stack
	current_state: u32,
	states:        [STATE_MAX]State,
	state:         State,

	// Transforms stack
	current_transform: u32,
	transforms:        [TRANSFORMS_MAX]Mat2x3,

	// configurable in Desc
	current_vertex: u32,
	vertices:       []Vertex,
	vertices_size:  u32,

	// configurable in Desc
	current_command: u32,
	commands:        []_Command,
	commands_size:   u32,

	// Desc Uniforms stack
	current_uniform: u32,
	uniforms:        []Uniform,
	uniforms_size:   u32,
}

@(private)
_gp: _Gp

// Map a blend mode to a dense pipeline cache slot (C indexes the cache by
// raw SDL_BlendMode values, which are sparse; dense slots avoid OOB).
@(private)
_blend_slot :: proc(blend_mode: BlendMode) -> int {
	switch blend_mode {
	case .NONE:
		return 0
	case .BLEND:
		return 1
	case .ADD:
		return 2
	case .MOD:
		return 3
	case .MUL:
		return 4
	case .BLEND_PREMULTIPLIED:
		return 5
	case .ADD_PREMULTIPLIED:
		return 6
	case .SIZE:
		return 0
	}
	return 0
}

@(private)
_pipeline_index :: proc(primitive_type: PrimitiveType, blend_mode: BlendMode) -> int {
	return int(primitive_type) * int(BlendMode.SIZE) + _blend_slot(blend_mode)
}

@(private)
_find_or_create_pipeline :: proc(primitive_type: PrimitiveType, blend_mode: BlendMode) -> Pipeline {
	index := _pipeline_index(primitive_type, blend_mode)
	pipeline := _gp.pipelines[index]

	if pipeline.id == INVALID_ID {
		pipeline = CreatePipeline(_gp.shader_vert, _gp.shader_frag, primitive_type, blend_mode)
		_gp.pipelines[index] = pipeline
	}

	return pipeline
}

// Setup painter context. Returns false if setup failed, use GetLastError()
// to get more information about the error.
Setup :: proc(desc: ^Desc) -> bool {
	assert(_gp.initialized == 0)
	assert(desc != nil)

	_last_error = .NONE

	_gp.initialized = _INIT_COOKIE

	_gp.desc.max_vertices = _VERTICES_MAX if desc.max_vertices == 0 else desc.max_vertices
	_gp.desc.max_commands = _COMMANDS_MAX if desc.max_commands == 0 else desc.max_commands
	_gp.desc.window = desc.window
	_gp.desc.gpu_device = desc.gpu_device

	_gp.vertices_size = _gp.desc.max_vertices
	_gp.commands_size = _gp.desc.max_commands
	_gp.uniforms_size = _gp.desc.max_commands
	_gp.vertices = make([]Vertex, int(_gp.vertices_size))
	_gp.commands = make([]_Command, int(_gp.commands_size))
	_gp.uniforms = make([]Uniform, int(_gp.uniforms_size))

	// Setup resources management for shaders, pipelines and images

	_ShaderSetup(_gp.desc.gpu_device)
	_PipelineSetup(_gp.desc.gpu_device, _gp.desc.window)
	if !_ImageSetup(_gp.desc.gpu_device, _gp.desc.window) {
		Shutdown()
		return false
	}

	// Create a white texture

	texture_format := sdl.GetGPUSwapchainTextureFormat(_img_ctx.gpu_device, _img_ctx.window)
	pixel_format := sdl.GetPixelFormatFromGPUTextureFormat(texture_format)
	format_details := sdl.GetPixelFormatDetails(pixel_format)

	white := sdl.MapRGBA(format_details, nil, 255, 255, 255, 255)
	white_pixels := [4]u32{white, white, white, white}

	white_surface := sdl.CreateSurfaceFrom(2, 2, pixel_format, raw_data(white_pixels[:]), c.int(format_details.bytes_per_pixel) * 2)
	if white_surface == nil {
		Shutdown()
		_set_error(.CREATE_WHITE_TEXTURE_FAILED)
		return false
	}
	defer sdl.DestroySurface(white_surface)

	_gp.white_image = CreateImage(white_surface)
	if _gp.white_image.id == INVALID_ID {
		Shutdown()
		return false
	}

	// Create a GPU transfer buffer for vertex data

	vertex_transfer_buffer_create_info := sdl.GPUTransferBufferCreateInfo{
		usage = .UPLOAD,
		size  = u32(_gp.desc.max_vertices) * u32(size_of(Vertex)),
	}

	_gp.vertex_transfer_buffer = sdl.CreateGPUTransferBuffer(desc.gpu_device, vertex_transfer_buffer_create_info)
	if _gp.vertex_transfer_buffer == nil {
		_set_error(.CREATE_TRANSFER_BUFFER_FAILED)
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
		_set_error(.CREATE_VERTEX_BUFFER_FAILED)
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

	vert, frag, shaders_ok := _create_common_shaders(desc.gpu_device)
	if !shaders_ok {
		return false
	}
	_gp.shader_vert = vert
	_gp.shader_frag = frag

	// Create common pipelines

	is_ok := true
	is_ok &= _find_or_create_pipeline(.POINTS, .NONE).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.POINTS, .BLEND).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.LINES, .NONE).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.LINES, .BLEND).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.LINE_STRIP, .NONE).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.LINE_STRIP, .BLEND).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.TRIANGLES, .NONE).id != INVALID_ID
	is_ok &= _find_or_create_pipeline(.TRIANGLES, .BLEND).id != INVALID_ID

	if !is_ok {
		_set_error(.CREATE_COMMON_PIPELINE_FAILED)
		Shutdown()
		return false
	}

	return true
}

// Shutdown painter context.
Shutdown :: proc() {
	if _gp.initialized != _INIT_COOKIE {
		return
	}

	// Destroy common pipelines

	for i := 0; i < len(_gp.pipelines); i += 1 {
		if _gp.pipelines[i].id != INVALID_ID {
			DestroyPipeline(_gp.pipelines[i])
			_gp.pipelines[i] = Pipeline{id = INVALID_ID}
		}
	}

	// Destroy common shader

	if _gp.shader_vert.id != INVALID_ID {
		DestroyShader(_gp.shader_vert)
		_gp.shader_vert = Shader{id = INVALID_ID}
	}

	if _gp.shader_frag.id != INVALID_ID {
		DestroyShader(_gp.shader_frag)
		_gp.shader_frag = Shader{id = INVALID_ID}
	}

	// Destroy nearest sampler

	if _gp.nearest_samplers != nil {
		sdl.ReleaseGPUSampler(_gp.desc.gpu_device, _gp.nearest_samplers)
		_gp.nearest_samplers = nil
	}

	// Destroy vertex data buffer

	if _gp.vertex_data_buffer != nil {
		sdl.ReleaseGPUBuffer(_gp.desc.gpu_device, _gp.vertex_data_buffer)
		_gp.vertex_data_buffer = nil
	}

	// Destroy vertex transfer buffer

	if _gp.vertex_transfer_buffer != nil {
		sdl.ReleaseGPUTransferBuffer(_gp.desc.gpu_device, _gp.vertex_transfer_buffer)
		_gp.vertex_transfer_buffer = nil
	}

	// Destroy white texture

	if _gp.white_image.id != INVALID_ID {
		DestroyImage(_gp.white_image)
		_gp.white_image = Image{id = INVALID_ID}
	}

	// Shutdown resources management for shaders, pipelines and images
	_ImageShutdown()
	_PipelineShutdown()
	_ShaderShutdown()

	delete(_gp.uniforms)
	delete(_gp.commands)
	delete(_gp.vertices)

	_gp = {}
}

// Begin recording draw calls for the current frame. This should be called
// after setting up the painter and acquiring a swapchain texture and command
// buffer for the current frame.
// If return false then an error occurred and the frame should be skipped,
// use GetLastError() to get more information about the error.
Begin :: proc(width, height: i32) -> bool {
	assert(_gp.initialized == _INIT_COOKIE)

	_gp.states[_gp.current_state] = _gp.state
	_gp.current_state += 1

	w := f32(width)
	h := f32(height)

	_gp.state.projection = Mat2x3{2.0 / w, 0.0, -1.0, 0.0, -2.0 / h, 1.0}
	_gp.state.transform = Mat2x3{1.0, 0.0, 0.0, 0.0, 1.0, 0.0}
	_gp.state.mvp = _gp.state.projection

	_gp.state.texture.count = 1
	_gp.state.texture.images[0] = _gp.white_image
	_gp.state.texture.samplers[0] = _gp.nearest_samplers

	invalid_image := Image{id = INVALID_ID}
	for i := 1; i < TEXTURE_SLOTS_MAX; i += 1 {
		_gp.state.texture.images[i] = invalid_image
		_gp.state.texture.samplers[i] = _gp.nearest_samplers
	}

	_gp.state.uniform = {}

	_gp.state.blend_mode = .NONE

	_gp.state.frame_size.w = width
	_gp.state.frame_size.h = height
	_gp.state.viewport = IRect{0, 0, width, height}
	_gp.state.scissor = IRect{0, 0, -1, -1}
	_gp.state.color = sdl.Color{255, 255, 255, 255}

	_gp.state.thickness = max(1.0 / w, 1.0 / h)
	_gp.state.base_vertex = _gp.current_vertex
	_gp.state.base_uniform = _gp.current_uniform
	_gp.state.base_command = _gp.current_command

	return true
}

// Flush the recorded draw calls to the GPU. Returns false if an error
// occurred, use GetLastError() to get more information about the error.
Flush :: proc(cmd_buffer: ^sdl.GPUCommandBuffer, texture: ^sdl.GPUTexture) -> bool {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(cmd_buffer != nil)
	assert(texture != nil)

	_ImageFlush(cmd_buffer)

	end_command := _gp.current_command
	end_vertex := _gp.current_vertex

	vertices_count := end_vertex - _gp.state.base_vertex // Number of vertices to draw

	// Rewind Index
	_gp.current_command = _gp.state.base_command
	_gp.current_uniform = _gp.state.base_uniform
	_gp.current_vertex = _gp.state.base_vertex

	// Error, Nothing to draw
	if _last_error != .NONE {
		return false
	}

	// Nothing to draw
	if end_command <= _gp.state.base_command {
		return true
	}

	vertex_data := sdl.MapGPUTransferBuffer(_gp.desc.gpu_device, _gp.vertex_transfer_buffer, true)
	if vertex_data == nil {
		_set_error(.FLUSH_FAILED)
		return false
	}

	mem.copy(vertex_data, &_gp.vertices[_gp.state.base_vertex], int(vertices_count) * size_of(Vertex))

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
		offset = _gp.state.base_vertex * u32(size_of(Vertex)),
		size   = vertices_count * u32(size_of(Vertex)),
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

	cur_pipeline_id: u32 = IMPOSSIBLE_ID
	cur_uniform_index: u32 = IMPOSSIBLE_ID
	cur_image_ids: [TEXTURE_SLOTS_MAX]u32
	for i := 0; i < TEXTURE_SLOTS_MAX; i += 1 {
		cur_image_ids[i] = IMPOSSIBLE_ID
	}

	// Flush commands
	for i := _gp.state.base_command; i < end_command; i += 1 {
		cmd := &_gp.commands[i]

		#partial switch cmd.cmd {
		case .DRAW:
			if vertices_count == 0 {
				break
			}

			draw := cmd.args.draw

			rebind_uniforms := false
			rebind_texture := false

			// Check if pipeline needs to be changed
			if draw.pipeline.id != cur_pipeline_id {
				cur_pipeline_id = draw.pipeline.id

				// Bind pipeline
				sdl.BindGPUGraphicsPipeline(render_pass, GetGPUPipeline(draw.pipeline))

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

			for j := 0; j < TEXTURE_SLOTS_MAX; j += 1 {
				image_id: u32 = INVALID_ID

				if j < int(draw.texture.count) {
					image_id = draw.texture.images[j].id
				}

				if image_id != cur_image_ids[j] {
					cur_image_ids[j] = image_id
					rebind_texture = true
				}

				if image_id != INVALID_ID {
					image_bindings[j] = sdl.GPUTextureSamplerBinding{
						texture = GetImageGPUTexture(draw.texture.images[j]),
						sampler = draw.texture.samplers[j],
					}
				} else {
					image_bindings[j] = sdl.GPUTextureSamplerBinding{
						texture = GetImageGPUTexture(_gp.white_image),
						sampler = _gp.nearest_samplers,
					}
				}
			}

			// Rebind textures if needed
			if rebind_texture {
				sdl.BindGPUFragmentSamplers(render_pass, 0, &image_bindings[0], TEXTURE_SLOTS_MAX)
			}

			// Rebind uniforms if needed
			if rebind_uniforms && cur_uniform_index != IMPOSSIBLE_ID {
				uniform := &_gp.uniforms[draw.uniform_index]

				if uniform.vs_size > 0 {
					sdl.PushGPUVertexUniformData(cmd_buffer, u32(UniformSlot.VS), rawptr(&uniform.data), u32(uniform.vs_size))
				}
				if uniform.fs_size > 0 {
					sdl.PushGPUFragmentUniformData(cmd_buffer, u32(UniformSlot.FS), rawptr(&uniform.data), u32(uniform.fs_size))
				}
			}

			vertex_buffer_binding := sdl.GPUBufferBinding{
				buffer = _gp.vertex_data_buffer,
				offset = draw.vertex_index * u32(size_of(Vertex)),
			}

			// In every case we need to bind vertex buffers
			sdl.BindGPUVertexBuffers(render_pass, 0, &vertex_buffer_binding, 1)

			sdl.DrawGPUPrimitives(render_pass, draw.vertices_count, 1, 0, 0)
		case .VIEWPORT:
			viewport_rect := cmd.args.viewport
			viewport := sdl.GPUViewport{
				x = f32(viewport_rect.x),
				y = f32(viewport_rect.y),
				w = f32(viewport_rect.w),
				h = f32(viewport_rect.h),
			}
			sdl.SetGPUViewport(render_pass, viewport)
		case .SCISSOR:
			scissor_rect := cmd.args.scissor
			scissor := sdl.Rect{
				x = c.int(scissor_rect.x),
				y = c.int(scissor_rect.y),
				w = c.int(scissor_rect.w),
				h = c.int(scissor_rect.h),
			}
			sdl.SetGPUScissor(render_pass, scissor)
		}
	}

	sdl.EndGPURenderPass(render_pass)

	return true
}

// End recording draw calls for the current frame.
End :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)

	_gp.current_state -= 1
	_gp.state = _gp.states[_gp.current_state]
}

// Painter (Private): batching internals
// ----------------------------------------------------------------------------

@(private)
_MOVE_VERTICES_MAX :: 96

@(private)
_next_uniform :: proc() -> ^Uniform {
	if _gp.current_uniform < u32(len(_gp.uniforms)) {
		uniform := &_gp.uniforms[_gp.current_uniform]
		_gp.current_uniform += 1
		return uniform
	} else {
		_set_error(.UNIFORMS_FULL)
		return nil
	}
}

@(private)
_prev_uniform :: proc() -> ^Uniform {
	if _gp.current_uniform > 0 {
		return &_gp.uniforms[_gp.current_uniform - 1]
	} else {
		return nil
	}
}

@(private)
_next_vertices :: proc(count: u32) -> [^]Vertex {
	if _gp.current_vertex + count <= u32(len(_gp.vertices)) {
		vertices := cast([^]Vertex)&_gp.vertices[_gp.current_vertex]
		_gp.current_vertex += count
		return vertices
	} else {
		_set_error(.VERTICES_FULL)
		return nil
	}
}

@(private)
_next_command :: proc() -> ^_Command {
	if _gp.current_command < u32(len(_gp.commands)) {
		cmd := &_gp.commands[_gp.current_command]
		_gp.current_command += 1
		return cmd
	} else {
		return nil
	}
}

@(private)
_prev_command :: proc(count: u32) -> ^_Command {
	if _gp.current_command - _gp.state.base_command >= count {
		return &_gp.commands[_gp.current_command - count]
	} else {
		return nil
	}
}

@(private)
_default_projection :: proc(width, height: i32) -> Mat2x3 {
	w := f32(width)
	h := f32(height)

	return create_mat2x3(2.0 / w, 0.0, -1.0, 0.0, -2.0 / h, 1.0)
}

@(private)
_mul_projection_transform :: proc(projection, transform: ^Mat2x3) -> Mat2x3 {
	x := projection.m00
	y := projection.m11

	out := Mat2x3{}

	out.m00 = x * transform.m00
	out.m01 = x * transform.m01
	out.m02 = x * transform.m02 + projection.m02

	out.m10 = y * transform.m10
	out.m11 = y * transform.m11
	out.m12 = y * transform.m12 + projection.m12

	return out
}

@(private)
_mat3_mul_vec2 :: proc(m: ^Mat2x3, v: ^Vec2) -> Vec2 {
	return Vec2{m.m00 * v.x + m.m01 * v.y + m.m02, m.m10 * v.x + m.m11 * v.y + m.m12}
}

@(private)
_transform :: proc(m: ^Mat2x3, dst, src: []Vec2) {
	assert(len(dst) >= len(src))
	for i in 0..<len(src) {
		dst[i] = _mat3_mul_vec2(m, &src[i])
	}
}

@(private)
_region_overlaps :: proc(a, b: _Region) -> bool {
	return !(a.x2 <= b.x1 || b.x2 <= a.x1 || a.y2 <= b.y1 || b.y2 <= a.y1)
}

@(private)
_merge_draw_commands :: proc(
	pipeline: Pipeline,
	texture: TextureUniform,
	uniform: ^Uniform,
	region: _Region,
	vertex_index: u32,
	vertices_count: u32,
) -> bool {
	vertices_count := vertices_count
	prev_cmd: ^_Command = nil
	inter_cmds: [OPTIMIZER_DEPTH]^_Command
	inter_cmd_count := 0

	// Find commands that are mergable
	lookup_depht := OPTIMIZER_DEPTH
	for depth := 0; depth <= lookup_depht; depth += 1 {
		cmd := _prev_command(u32(depth) + 1)

		if cmd == nil {
			break // Stop on nonexistent command
		}

		if cmd.cmd == .NONE {
			lookup_depht += 1
			continue // Command was optimized, continue looking
		}

		if cmd.cmd != .DRAW {
			break // Stop on scissor or viewport
		}

		// Only command with the same pipeline, texture and uniform can be merged
		texture_bytes := mem.any_to_bytes(texture)
		cmd_texture_bytes := mem.any_to_bytes(cmd.args.draw.texture)
		uniform_match := uniform == nil
		if !uniform_match {
			uniform_match = mem.compare(mem.any_to_bytes(uniform^), mem.any_to_bytes(_gp.uniforms[cmd.args.draw.uniform_index])) == 0
		}
		if cmd.args.draw.pipeline.id == pipeline.id &&
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
	for i := 0; i < inter_cmd_count; i += 1 {
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

	if !overlaps_next { 	// Merge with the previous draw command
		if inter_cmd_count > 0 {
			// Can't merge if we don't have enough space for vertices
			if _gp.current_vertex + vertices_count > u32(len(_gp.vertices)) {
				return false
			}

			prev_end_vertex := prev_cmd.args.draw.vertex_index + prev_cmd.args.draw.vertices_count
			prev_vertices_count := _gp.current_vertex - prev_end_vertex

			// Avoid moving too meny vertices, otherwise it can cause performance
			// regression
			if prev_vertices_count > _MOVE_VERTICES_MAX {
				return false
			}

			// Re-organized vertices
			mem.copy(&_gp.vertices[prev_end_vertex + vertices_count], &_gp.vertices[prev_end_vertex], int(prev_vertices_count) * size_of(Vertex))
			mem.copy_non_overlapping(&_gp.vertices[prev_end_vertex], &_gp.vertices[vertex_index + vertices_count], int(vertices_count) * size_of(Vertex))

			// Offset vertices of inter_cmds
			for i := 0; i < inter_cmd_count; i += 1 {
				inter_cmds[i].args.draw.vertex_index += vertices_count
			}
		}

		// Update draw region and vertices
		prev_region.x1 = min(prev_region.x1, region.x1)
		prev_region.y1 = min(prev_region.y1, region.y1)
		prev_region.x2 = max(prev_region.x2, region.x2)
		prev_region.y2 = max(prev_region.y2, region.y2)
		prev_cmd.args.draw.vertices_count += vertices_count
		prev_cmd.args.draw.region = prev_region
	} else { 	// Merge with the next draw command
		assert(inter_cmd_count > 0)

		// Append new draw command
		cmd := _next_command()
		if cmd == nil {
			return false
		}

		prev_vertices_count := prev_cmd.args.draw.vertices_count

		// Can't merge if we don't have enough space for vertices
		if _gp.current_vertex + vertices_count > u32(len(_gp.vertices)) {
			return false
		}

		// Avoid moving too meny vertices, otherwise it can cause performance
		// regression
		if prev_vertices_count > _MOVE_VERTICES_MAX {
			return false
		}

		// Re-organized vertices
		mem.copy(&_gp.vertices[vertex_index + prev_vertices_count], &_gp.vertices[vertex_index], int(vertices_count) * size_of(Vertex))
		mem.copy_non_overlapping(&_gp.vertices[vertex_index], &_gp.vertices[prev_cmd.args.draw.vertex_index], int(prev_vertices_count) * size_of(Vertex))

		// Update draw region and vertices
		prev_region.x1 = min(prev_region.x1, region.x1)
		prev_region.y1 = min(prev_region.y1, region.y1)
		prev_region.x2 = max(prev_region.x2, region.x2)
		prev_region.y2 = max(prev_region.y2, region.y2)
		_gp.current_vertex += prev_vertices_count
		vertices_count += prev_vertices_count

		// Configure the new draw command
		cmd.cmd = .DRAW
		cmd.args.draw.pipeline = pipeline
		cmd.args.draw.texture = texture
		cmd.args.draw.region = prev_region
		cmd.args.draw.uniform_index = prev_cmd.args.draw.uniform_index
		cmd.args.draw.vertex_index = vertex_index
		cmd.args.draw.vertices_count = vertices_count

		// Force skipping the previous draw command
		prev_cmd.cmd = .NONE
	}
	return true
}

@(private)
_queue_draw :: proc(pipeline: Pipeline, region: _Region, vertex_index: u32, vertices_count: u32, primitive_type: PrimitiveType) {
	pipeline := pipeline
	uniform: ^Uniform = nil
	if _gp.state.pipeline.id != INVALID_ID {
		pipeline = _gp.state.pipeline
		uniform = &_gp.state.uniform
	}

	// If the region is completely outside of the viewport, skip the draw call
	if region.x1 > 1.0 || region.y1 > 1.0 || region.x2 < -1.0 || region.y2 < -1.0 {
		_gp.current_vertex -= vertices_count // rollback allocated vertices
		return
	}

	// Try to merge with previous draw command
	if primitive_type != .TRIANGLE_STRIP &&
	   primitive_type != .LINE_STRIP &&
	   _merge_draw_commands(pipeline, _gp.state.texture, uniform, region, vertex_index, vertices_count) {
		return
	}

	// Try to reuse previous uniform if possible
	uniform_index := u32(IMPOSSIBLE_ID)
	if uniform != nil {
		prev_uniform := _prev_uniform()

		reuse_uniform := prev_uniform != nil && mem.compare(mem.any_to_bytes(prev_uniform^), mem.any_to_bytes(uniform^)) == 0

		if !reuse_uniform {
			next_uniform := _next_uniform()
			if next_uniform == nil {
				_gp.current_vertex -= vertices_count // rollback allocated vertices
				return
			}
			next_uniform^ = _gp.state.uniform
		}

		uniform_index = _gp.current_uniform - 1 // - 1 since _bxr_painter_next_uniform
		// already incremented the index
	}

	// New draw command
	cmd := _next_command()

	if cmd == nil {
		_gp.current_vertex -= vertices_count // rollback allocated vertices
		return
	}

	cmd.cmd = .DRAW
	cmd.args.draw.pipeline = pipeline
	cmd.args.draw.texture = _gp.state.texture
	cmd.args.draw.region = region
	cmd.args.draw.uniform_index = uniform_index
	cmd.args.draw.vertex_index = vertex_index
	cmd.args.draw.vertices_count = vertices_count
}

@(private)
_draw_solid :: proc(primitive_type: PrimitiveType, vertices: [^]Vec2, vertices_count: u32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	if vertices_count == 0 {
		return
	}

	// Setup vertices
	vertex_index := _gp.current_vertex
	v := _next_vertices(vertices_count)
	if v == nil {
		return
	}

	thickness: f32 = 1.0
	if primitive_type == .POINTS || primitive_type == .LINES || primitive_type == .LINE_STRIP {
		thickness = _gp.state.thickness
	}
	color := _gp.state.color
	mvp := _gp.state.mvp
	region := _Region{max(f32), max(f32), -max(f32), -max(f32)}

	for i: u32 = 0; i < vertices_count; i += 1 {
		p := _mat3_mul_vec2(&mvp, &vertices[i])

		region.x1 = min(region.x1, p.x - thickness)
		region.y1 = min(region.y1, p.y - thickness)
		region.x2 = max(region.x2, p.x + thickness)
		region.y2 = max(region.y2, p.y + thickness)

		v[i].position = p
		v[i].texcoord = Vec2{0.0, 0.0}
		v[i].color = color
	}

	pipeline := _find_or_create_pipeline(primitive_type, _gp.state.blend_mode)

	// Queue draw
	_queue_draw(pipeline, region, vertex_index, vertices_count, primitive_type)
}

// Set the coordinate space boundaries in the current viewport.
SetProjection :: proc(left, right, bottom, top: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	width := right - left
	height := top - bottom

	_gp.state.projection = create_mat2x3(2.0 / width, 0.0, -(right + left) / width, 0.0, 2.0 / height, -(top + bottom) / height)

	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Reset the projection to the default coordinate space, which is the
// coordinate of the current viewport.
ResetProjection :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.projection = _default_projection(_gp.state.viewport.w, _gp.state.viewport.h)

	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Save the current transform matrix on the transform stack. To be pop later
// with PopTransform.
PushTransform :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(_gp.current_transform < TRANSFORMS_MAX)

	_gp.transforms[_gp.current_transform] = _gp.state.transform
	_gp.current_transform += 1
}

// Restore the transform matrix from the top of the transform stack.
PopTransform :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(_gp.current_transform > 0)

	_gp.current_transform -= 1
	_gp.state.transform = _gp.transforms[_gp.current_transform]
	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Set the current transform matrix to identity (no transformation).
ResetTransform :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.transform = create_mat2x3_identity()
	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Translates the 2D coordinates space.
Translate :: proc(x, y: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	// multiply by translate matrix:
	// 1.0f, 0.0f, tx,
	// 0.0f, 1.0f, ty,

	_gp.state.transform.m02 += x * _gp.state.transform.m00 + y * _gp.state.transform.m01
	_gp.state.transform.m12 += x * _gp.state.transform.m10 + y * _gp.state.transform.m11

	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Rotates the 2D coordinate space around the origin.
Rotate :: proc(angle: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	c := math.cos(angle)
	s := math.sin(angle)

	// Multiply by rotation matrix:
	//   c,   -s, 0.0f,
	//   s,    c, 0.0f,

	rotation := create_mat2x3(
		c * _gp.state.transform.m00 + s * _gp.state.transform.m01,
		-s * _gp.state.transform.m00 + c * _gp.state.transform.m01,
		_gp.state.transform.m02,
		c * _gp.state.transform.m10 + s * _gp.state.transform.m11,
		-s * _gp.state.transform.m10 + c * _gp.state.transform.m11,
		_gp.state.transform.m12,
	)

	_gp.state.transform = rotation
	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Rotates the 2D coordinate space around a point.
RotateAt :: proc(angle, ax, ay: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	Translate(ax, ay)
	Rotate(angle)
	Translate(-ax, -ay)
}

// Scales the 2D coordinate space around the origin.
Scale :: proc(sx, sy: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	// Multiply by scale matrix:
	//   sx, 0.0f, 0.0f,
	// 0.0f,   sy, 0.0f,

	_gp.state.transform.m00 *= sx
	_gp.state.transform.m01 *= sy
	_gp.state.transform.m10 *= sx
	_gp.state.transform.m11 *= sy

	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Scales the 2D coordinate space around a point.
ScaleAt :: proc(sx, sy, ax, ay: f32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	Translate(ax, ay)
	Scale(sx, sy)
	Translate(-ax, -ay)
}

// Set the current graphics pipeline.
SetPipeline :: proc(pipeline: Pipeline) {
	assert(_gp.initialized == _INIT_COOKIE)

	_gp.state.pipeline = pipeline

	// Reset uniforms when pipeline changes
	_gp.state.uniform = {}
}

// Reset the graphics pipeline to the default pipeline builtin pipeline.
ResetPipeline :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)

	pipeline := Pipeline{id = INVALID_ID}

	SetPipeline(pipeline)
}

// Set uniform data for the current pipeline.
SetUniform :: proc(vs_data: rawptr, vs_size: i32, fs_data: rawptr, fs_size: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.state.pipeline.id != INVALID_ID)

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
		mem.set(mem.ptr_offset(cast([^]u8)&_gp.state.uniform.data, size), 0, old_size - size)
	}

	_gp.state.uniform.vs_size = u16(vs_size)
	_gp.state.uniform.fs_size = u16(fs_size)
}

// Reset uniform data to the default state (current state color).
ResetUniform :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.state.pipeline.id != INVALID_ID)

	SetUniform(nil, 0, nil, 0)
}

// Set the current blend mode.
SetBlendMode :: proc(blend_mode: BlendMode) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.blend_mode = blend_mode
}

// Reset the current blend mode to the default blend mode (no blending).
ResetBlendMode :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.blend_mode = .NONE
}

// Sets current color.
SetColor :: proc(color: sdl.Color) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.color = color
}

// Gets current color.
GetColor :: proc() -> sdl.Color {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	return _gp.state.color
}

// Reset current color to the default color (white).
ResetColor :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.color = sdl.Color{255, 255, 255, 255}
}

// Sets current bound image in a texture channel.
SetImage :: proc(channel: i32, image: Image) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	ch := int(channel)
	if _gp.state.texture.images[ch].id == image.id {
		return
	}

	_gp.state.texture.images[ch] = image

	// Recalculate texture count
	texture_count := int(_gp.state.texture.count)
	for i := max(ch, texture_count - 1); i >= 0; i -= 1 {
		if _gp.state.texture.images[i].id != INVALID_ID {
			texture_count = i + 1
			break
		}
	}

	_gp.state.texture.count = u32(texture_count)
}

// Reset current bound image in a texture channel to the default (white
// texture).
ResetImage :: proc(channel: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	SetImage(channel, _gp.white_image)
}

// Remove current bound image from a texture channel (no texture).
UnsetImage :: proc(channel: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	SetImage(channel, Image{id = INVALID_ID})
}

// Set current bound sampler in a texture channel.
SetSampler :: proc(channel: i32, sampler: ^sdl.GPUSampler) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[int(channel)] = sampler
}

// Remove current bound sampler from a texture channel (no sampler).
UnsetSampler :: proc(channel: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[int(channel)] = nil
}

// Reset current bound sampler in a texture channel to default (nearest
// sampler).
ResetSampler :: proc(channel: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	_gp.state.texture.samplers[int(channel)] = _gp.nearest_samplers
}

// Set the screen are to draw to.
Viewport :: proc(x, y, w, h: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	// If no change in viewport, skip
	if _gp.state.viewport.x == x && _gp.state.viewport.y == y && _gp.state.viewport.w == w && _gp.state.viewport.h == h {
		return
	}

	// Try to reuse previous command
	cmd := _prev_command(1)
	if cmd != nil && cmd.cmd != .VIEWPORT {
		cmd = _next_command()
	}
	if cmd == nil {
		return
	}

	viewport := IRect{x = x, y = y, w = w, h = h}

	cmd.cmd = .VIEWPORT
	cmd.args.viewport = viewport

	// When viewport changes, scissor needs to be updated to keep the same region
	if !(_gp.state.scissor.w < 0 && _gp.state.scissor.h < 0) {
		_gp.state.scissor.x += x - _gp.state.viewport.x
		_gp.state.scissor.y += y - _gp.state.viewport.y
	}

	_gp.state.viewport = viewport
	_gp.state.thickness = max(1.0 / f32(w), 1.0 / f32(h))
	_gp.state.projection = _default_projection(w, h)
	_gp.state.mvp = _mul_projection_transform(&_gp.state.projection, &_gp.state.transform)
}

// Reset the viewport to default (0, 0, width, height).
ResetViewport :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	Viewport(0, 0, _gp.state.frame_size.w, _gp.state.frame_size.h)
}

// Set the clipping rectangle in the viewport.
Scissor :: proc(x, y, w, h: i32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	// Skip if scissor is the same
	if _gp.state.scissor.x == x && _gp.state.scissor.y == y && _gp.state.scissor.w == w && _gp.state.scissor.h == h {
		return
	}

	// Try to reuse previous command
	cmd := _prev_command(1)
	if cmd != nil && cmd.cmd != .SCISSOR {
		cmd = _next_command()
	}
	if cmd == nil {
		return
	}

	// Coordinates scissor relative to viewport
	viewport_scissor := IRect{
		x = _gp.state.viewport.x + x,
		y = _gp.state.viewport.y + y,
		w = w,
		h = h,
	}

	// Reset scissor
	if w < 0 && h < 0 {
		viewport_scissor.x = 0
		viewport_scissor.y = 0
		viewport_scissor.w = _gp.state.frame_size.w
		viewport_scissor.h = _gp.state.frame_size.h
	}

	cmd.cmd = .SCISSOR
	cmd.args.scissor = viewport_scissor

	_gp.state.scissor = IRect{x = x, y = y, w = w, h = h}
}

// Reset the clipping rectangle to default (viewport bounds).
ResetScissor :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	_gp.state.scissor = IRect{x = 0, y = 0, w = -1, h = -1}
}

// Reset all state to default.
ResetState :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	ResetViewport()
	ResetScissor()
	ResetProjection()
	ResetTransform()
	ResetBlendMode()
	ResetColor()
	ResetUniform()
	ResetPipeline()
}

// Clear the current viewport with the current color.
Clear :: proc() {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	// Setup vertices
	vertices_count := u32(6)
	vertex_index := _gp.current_vertex

	v := _next_vertices(vertices_count)
	if v == nil {
		return
	}

	// Compute vertices
	quad := [4]Vec2{
		{-1.0, -1.0}, // bottom-left
		{1.0, -1.0}, // bottom-right
		{1.0, 1.0}, // top-right
		{-1.0, 1.0}, // top-left
	}

	texcoord := Vec2{0.0, 0.0}
	color := _gp.state.color

	v[0] = Vertex{position = quad[0], texcoord = texcoord, color = color}
	v[1] = Vertex{position = quad[1], texcoord = texcoord, color = color}
	v[2] = Vertex{position = quad[2], texcoord = texcoord, color = color}
	v[3] = Vertex{position = quad[2], texcoord = texcoord, color = color}
	v[4] = Vertex{position = quad[3], texcoord = texcoord, color = color}
	v[5] = Vertex{position = quad[0], texcoord = texcoord, color = color}

	region := _Region{-1.0, -1.0, 1.0, 1.0}

	pipeline := _find_or_create_pipeline(.TRIANGLES, _gp.state.blend_mode)

	_queue_draw(pipeline, region, vertex_index, vertices_count, .TRIANGLES)
}

// Draw any primitive.
Draw :: proc(primitive_type: PrimitiveType, vertices: [^]Vertex, #any_int vertices_count: u32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	if vertices_count == 0 {
		return
	}

	// Setup vertices
	vertex_index := _gp.current_vertex
	v := _next_vertices(vertices_count)
	if v == nil {
		return
	}

	thickness: f32 = 1.0
	if primitive_type == .POINTS || primitive_type == .LINES || primitive_type == .LINE_STRIP {
		thickness = _gp.state.thickness
	}
	mvp := _gp.state.mvp
	region := _Region{max(f32), max(f32), -max(f32), -max(f32)}

	for i: u32 = 0; i < vertices_count; i += 1 {
		p := _mat3_mul_vec2(&mvp, &vertices[i].position)

		region.x1 = min(region.x1, p.x - thickness)
		region.y1 = min(region.y1, p.y - thickness)
		region.x2 = max(region.x2, p.x + thickness)
		region.y2 = max(region.y2, p.y + thickness)

		v[i].position = p
		v[i].texcoord = vertices[i].texcoord
		v[i].color = vertices[i].color
	}

	pipeline := _find_or_create_pipeline(primitive_type, _gp.state.blend_mode)

	// Queue draw
	_queue_draw(pipeline, region, vertex_index, vertices_count, primitive_type)
}

// Draw points in batch.
DrawPoints :: proc(points: [^]Point, #any_int count: u32) {
	_draw_solid(.POINTS, points, count)
}

// Draw a single point.
DrawPoint :: proc(point: Point) {
	p := point
	DrawPoints(([^]Point)(&p), 1)
}

// Draw lines in batch.
DrawLines :: proc(lines: [^]Line, #any_int count: u32) {
	_draw_solid(.LINES, cast([^]Vec2)lines, count * 2)
}

// Draw a single line.
DrawLine :: proc(line: Line) {
	l := line
	DrawLines(([^]Line)(&l), 1)
}

// Draw a stip of lines.
DrawLinesStrip :: proc(points: [^]Vec2, #any_int count: u32) {
	_draw_solid(.LINE_STRIP, points, count)
}

// Draw triangles in batch.
DrawFilledTriangles :: proc(triangles: [^]Triangle, #any_int count: u32) {
	_draw_solid(.TRIANGLES, cast([^]Vec2)triangles, count * 3)
}

// Draw a single triangle.
DrawFilledTriangle :: proc(triangle: Triangle) {
	t := triangle
	DrawFilledTriangles(([^]Triangle)(&t), 1)
}

// Draw a strip of triangles.
DrawFilledTrianglesStrip :: proc(points: [^]Vec2, #any_int count: u32) {
	_draw_solid(.TRIANGLE_STRIP, points, count)
}

// Draw rectangles in batch.
DrawFilledRects :: proc(rects: [^]Rect, count: u32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)

	if count == 0 {
		return
	}

	// Setup vertices
	total_vertices := count * 6 // 2 triangles per rect, 3 vertices each
	vertex_index := _gp.current_vertex
	v := _next_vertices(total_vertices)
	if v == nil {
		return
	}

	// Compute vertices
	color := _gp.state.color
	mvp := _gp.state.mvp
	region := _Region{max(f32), max(f32), -max(f32), -max(f32)}

	for i: u32 = 0; i < count; i += 1 {
		rect := &rects[i]
		quad := [4]Vec2{
			{rect.x, rect.y + rect.h}, // bottom-left
			{rect.x + rect.w, rect.y + rect.h}, // bottom-right
			{rect.x + rect.w, rect.y}, // top-right
			{rect.x, rect.y}, // top-left
		}

		_transform(&mvp, quad[:], quad[:])

		for j in 0..<4 {
			region.x1 = min(region.x1, quad[j].x)
			region.y1 = min(region.y1, quad[j].y)
			region.x2 = max(region.x2, quad[j].x)
			region.y2 = max(region.y2, quad[j].y)
		}

		texcoords := [4]Vec2{
			{0.0, 1.0}, // bottom-left
			{1.0, 1.0}, // bottom-right
			{1.0, 0.0}, // top-right
			{0.0, 0.0}, // top-left
		}

		// Make two triangles to form the quad
		o := i * 6
		v[o + 0] = Vertex{position = quad[0], texcoord = texcoords[0], color = color}
		v[o + 1] = Vertex{position = quad[1], texcoord = texcoords[1], color = color}
		v[o + 2] = Vertex{position = quad[2], texcoord = texcoords[2], color = color}
		v[o + 3] = Vertex{position = quad[3], texcoord = texcoords[3], color = color}
		v[o + 4] = Vertex{position = quad[0], texcoord = texcoords[0], color = color}
		v[o + 5] = Vertex{position = quad[2], texcoord = texcoords[2], color = color}
	}

	// Queue draw
	pipeline := _find_or_create_pipeline(.TRIANGLES, _gp.state.blend_mode)

	_queue_draw(pipeline, region, vertex_index, total_vertices, .TRIANGLES)
}

// Draw a single rectangle.
DrawFilledRect :: proc(rect: Rect) {
	r := rect
	DrawFilledRects(([^]Rect)(&r), 1)
}

// Draw textured rectangles in batch.
DrawTexturedRects :: proc(channel: i32, rects: [^]TexturedRect, #any_int count: u32) {
	assert(_gp.initialized == _INIT_COOKIE)
	assert(_gp.current_state > 0)
	assert(channel >= 0 && channel < TEXTURE_SLOTS_MAX)

	if count == 0 {
		return
	}

	// Setup vertices
	total_vertices := count * 6 // 2 triangles per rect, 3 vertices each
	vertex_index := _gp.current_vertex
	vertices := _next_vertices(total_vertices)
	if vertices == nil {
		return
	}

	// Get image info
	image := _gp.state.texture.images[int(channel)]

	width := GetImageWidth(image)
	height := GetImageHeight(image)

	// Check image dimension with unlikely

	iw := 1.0 / f32(width)
	ih := 1.0 / f32(height)

	// Compute vertices
	mvp := _gp.state.mvp
	color := _gp.state.color
	region := _Region{max(f32), max(f32), -max(f32), -max(f32)}

	for i: u32 = 0; i < count; i += 1 {
		quad := [4]Vec2{
			{rects[i].dst.x, rects[i].dst.y + rects[i].dst.h}, // bottom left
			{rects[i].dst.x + rects[i].dst.w, rects[i].dst.y + rects[i].dst.h}, // bottom right
			{rects[i].dst.x + rects[i].dst.w, rects[i].dst.y}, // top right
			{rects[i].dst.x, rects[i].dst.y}, // top left
		}

		_transform(&mvp, quad[:], quad[:])

		for j in 0..<4 {
			region.x1 = min(region.x1, quad[j].x)
			region.y1 = min(region.y1, quad[j].y)
			region.x2 = max(region.x2, quad[j].x)
			region.y2 = max(region.y2, quad[j].y)
		}

		tl := rects[i].src.x * iw
		tt := rects[i].src.y * ih
		tr := (rects[i].src.x + rects[i].src.w) * iw
		tb := (rects[i].src.y + rects[i].src.h) * ih

		vtexquad := [4]Vec2{
			{tl, tb}, // bottom-left
			{tr, tb}, // bottom-right
			{tr, tt}, // top-right
			{tl, tt}, // top-left
		}

		o := i * 6
		v := cast([^]Vertex)&vertices[o]
		v[0] = Vertex{position = quad[0], texcoord = vtexquad[0], color = color}
		v[1] = Vertex{position = quad[1], texcoord = vtexquad[1], color = color}
		v[2] = Vertex{position = quad[2], texcoord = vtexquad[2], color = color}
		v[3] = Vertex{position = quad[3], texcoord = vtexquad[3], color = color}
		v[4] = Vertex{position = quad[0], texcoord = vtexquad[0], color = color}
		v[5] = Vertex{position = quad[2], texcoord = vtexquad[2], color = color}
	}

	// Queue draw
	pipeline := _find_or_create_pipeline(.TRIANGLES, _gp.state.blend_mode)

	_queue_draw(pipeline, region, vertex_index, total_vertices, .TRIANGLES)
}

// Draw a single textured rectangle.
DrawTexturedRect :: proc(channel: i32, rect: TexturedRect) {
	r := rect
	DrawTexturedRects(channel, ([^]TexturedRect)(&r), 1)
}
