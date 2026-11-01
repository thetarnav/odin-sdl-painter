package example

import "core:c"
import "core:log"
import sdl "vendor:sdl3"
import gp ".."

RENDER_TEXTURE_FRAMES :: 5

_render_textures:  [RENDER_TEXTURE_FRAMES]gp.Render_Texture
_render_source:     gp.Image
_render_cmd_buffer: ^sdl.GPUCommandBuffer

sample_render_texture_setup :: proc () {
	surface := sdl.LoadSurface(#directory + "../images/hello-world.png")
	if surface == nil {
		log.errorf("Failed to load image: %s", sdl.GetError())
	}
	_render_source = gp.make_image_from_surface(surface) or_else panic(gp.get_error_message(gp.get_last_error()))
	sdl.DestroySurface(surface)
}

// Each recursion level renders into its own texture.
sample_render_texture_get_render_texture :: proc (level: int) -> gp.Render_Texture {
	assert(level >= 0 && level < RENDER_TEXTURE_FRAMES)

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window_size := gp.Vec2i{i32(window_width), i32(window_height)}

	if window_size != gp.get_render_texture_size(_render_textures[level]) {
		gp.destroy_render_texture(_render_textures[level])
		_render_textures[level] = gp.make_render_texture(window_size) or_else panic(gp.get_error_message(gp.get_last_error()))
	}

	return _render_textures[level]
}

sample_render_texture_render :: proc (dt: u64, cmd_buffer: ^sdl.GPUCommandBuffer) {
	_render_cmd_buffer = cmd_buffer

	gp.set_blend_mode(.Blend)

	gp.set_color({90, 10, 160, 255})
	gp.clear()

	draw_nested_frame(0)

	gp.translate(60, 20)

	draw_nested_frame(0)
}

sample_render_texture_shutdown :: proc () {
	gp.destroy_image(_render_source)
	for texture in _render_textures {
		gp.destroy_render_texture(texture)
	}
}

draw_nested_frame :: proc (frame_idx: int) {

	if frame_idx >= RENDER_TEXTURE_FRAMES do return

	gp.transform_push()
	defer gp.transform_pop()

	gp.translate(10, 30)
	opacity := f32(RENDER_TEXTURE_FRAMES-frame_idx) / RENDER_TEXTURE_FRAMES

	cmd_buffer := _render_cmd_buffer
	assert(cmd_buffer != nil)

	render_texture := sample_render_texture_get_render_texture(frame_idx)

	if !gp.begin_render_texture(cmd_buffer, render_texture) {
		log.errorf("nested begin error: %s", gp.get_error_message(gp.get_last_error()))
		return
	}
	gp.set_color({0, 0, 0, 0})
	gp.clear()

	gp.set_color({214, 48, 49, 255})
	gp.draw_rect_xywh(16, 16, 96, 48)

	gp.set_color({46, 204, 113, 255})
	gp.draw_triangle({
		{32, 104},
		{96, 104},
		{64, 56},
	})

	gp.set_color({52, 152, 219, 255})
	gp.draw_line({120, 24}, {232, 120})

	gp.set_color({241, 196, 15, 255})
	gp.draw_line_strip({
		{120, 48},
		{160, 104},
		{200, 48},
		{240, 104},
	})

	gp.set_color({255, 255, 255, 255})
	gp.draw_points({
		{16, 120},
		{32, 120},
		{48, 120},
		{64, 120},
	})

	gp.set_image(0, _render_source)
	gp.set_color(255)
	gp.draw_textured_rect(0, {
		dst = {{150, 16}, Vec2(gp.get_image_size(_render_source))},
		src = {0, Vec2(gp.get_image_size(_render_source))},
	})
	gp.reset_image(0)

	draw_nested_frame(frame_idx+1)

	if !gp.end_render_texture() {
		log.errorf("nested frame error: %s", gp.get_error_message(gp.get_last_error()))
		return
	}

	// Copy from render texture to swapchain
	gp.set_color({255, 255, 255, u8(opacity * 255)})
	gp.set_image(0, render_texture)
	gp.draw_textured_rect(0, {
		dst = {0, Vec2(gp.get_image_size(render_texture))},
		src = {0, Vec2(gp.get_image_size(render_texture))},
	})
	gp.reset_image(0)
}
