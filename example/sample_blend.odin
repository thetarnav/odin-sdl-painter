package example

import "core:c"
import sdl "vendor:sdl3"
import gp ".."

draw_rects :: proc (brightness, alpha: u8) {
	// Red rectangle
	gp.set_color({brightness, 0, 0, alpha})
	gp.draw_rect(gp.Rect{{10, 10}, {50, 50}})

	gp.translate(10, 10)

	// Green rectangle
	gp.set_color({0, brightness, 0, alpha})
	gp.draw_rect(gp.Rect{{10, 10}, {50, 50}})

	gp.translate(10, 10)

	// Blue rectangle
	gp.set_color({0, 0, brightness, alpha})
	gp.draw_rect(gp.Rect{{10, 10}, {50, 50}})
}

draw_checkboard :: proc (width, height: int) {
	size := 20
	for y := 0; y < height; y += size {
		for x := 0; x < width; x += size {
			is_white := ((x / size) + (y / size)) % 2 == 0
			color := is_white ? \
				sdl.Color{150, 150, 150, 255} : \
				sdl.Color{50, 50, 50, 255}
			gp.set_color(color)
			gp.draw_rect(gp.Rect{{f32(x), f32(y)}, {f32(size), f32(size)}})
		}
	}
}

sample_blend_setup :: proc () {}

sample_blend_render :: proc (delta_time_ms: u64) {
	brightness := u8(255)
	alpha      := u8(128)

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window := [2]int{int(window_width), int(window_height)}

	gp.set_color({0,0,0,255})
	gp.clear()

	gp.blend_mode_set(.None)

	draw_checkboard(**window)

	gp.blend_mode_set(.None)
	if gp.transform_scope() {
		gp.translate(0, 0)
		draw_rects(brightness, alpha)
	}

	gp.blend_mode_set(.Blend)
	gp.push_transform()
	gp.translate(80, 0)
	draw_rects(brightness, alpha)
	gp.pop_transform()

	gp.blend_mode_set(.Blend_Premultiplied)
	gp.push_transform()
	gp.translate(160, 0)
	draw_rects(brightness, alpha)
	gp.pop_transform()

	gp.blend_mode_set(.Add)
	gp.push_transform()
	gp.translate(80, 80)
	draw_rects(brightness, alpha)
	gp.pop_transform()

	gp.blend_mode_set(.Add_Premultiplied)
	gp.push_transform()
	gp.translate(160, 80)
	draw_rects(brightness, alpha)
	gp.pop_transform()

	gp.blend_mode_set(.Mod)
	gp.push_transform()
	gp.translate(80, 160)
	draw_rects(brightness, alpha)
	gp.pop_transform()

	gp.blend_mode_set(.Mul)
	gp.push_transform()
	gp.translate(160, 160)
	draw_rects(brightness, alpha)
	gp.pop_transform()
}

sample_blend_shutdown :: proc () {}
