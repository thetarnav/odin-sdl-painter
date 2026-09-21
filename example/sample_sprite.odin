package example

import "core:c"
import "core:log"
import "core:math/rand"
import sdl "vendor:sdl3"
import gp ".."

image_sprite: gp.Image

sample_sprite_setup :: proc () {
	surface := sdl.LoadSurface(#directory+"../images/sprites.png")
	if surface == nil {
		log.errorf("Failed to load image: %s", sdl.GetError())
	}

	image_sprite = gp.make_image(surface)

	sdl.DestroySurface(surface)
}

sample_sprite_render :: proc (delta_time_ms: u64) {

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window := Vec2i{int(window_width), int(window_height)}

	tile_region := [3]Rect{
		{{0,  0}, {32, 32}}, // tile 1
		{{32, 0}, {32, 32}}, // tile 2
		{{64, 0}, {32, 32}}, // tile 3
	}

	gp.set_blend_mode(.Blend)
	gp.set_color(255)
	gp.set_image(0, image_sprite)

	for i in 0..<4096 {
		x := rand.int_max(window.x)
		y := rand.int_max(window.y)

		src_rect := tile_region[i % 3]
		dst_rect := Rect{{f32(x), f32(y)}, 32 * 2}

		gp.draw_textured_rect(0, gp.Textured_Rect{dst_rect, src_rect})
	}
}

sample_sprite_shutdown :: proc () {
	gp.destroy_image(image_sprite)
}
