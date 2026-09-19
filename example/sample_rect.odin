package example

import "core:c"
import sdl "vendor:sdl3"
import gp ".."

image_rect: gp.Image

sample_rect_setup :: proc () {
	surface := sdl.LoadSurface(#directory+"../images/hello-world.png")
	if surface == nil {
		sdl.Log("Failed to load image: %s", sdl.GetError())
	}

	image_rect = gp.create_image(surface)

	sdl.DestroySurface(surface)
}

sample_rect_render :: proc (delta_time_ms: u64) {

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window := [2]int{int(window_width), int(window_height)}

	h := window/2

	// Draw a red filled rectangle.
	{
		gp.set_viewport(0, 0, i32(h.x), window_height)
		gp.set_color({10, 10, 10, 255})
		gp.clear()

		gp.push_transform()
		{
			gp.set_color({255, 0, 0, 255})

			// Move to the left area of the viewport
			gp.translate(f32(h.x) * 0.5, f32(h.y))

			half_shape := f32(window.x) * 0.15 // 15% of the viewport width

			gp.draw_rect(gp.Rect_Vec2{{-half_shape,-half_shape},{half_shape*2,half_shape*2}})
		}
		gp.pop_transform()
	}

	// Draw a textured rectangle keeping it's original color.
	{
		gp.set_viewport(i32(h.x), 0, i32(h.x), window_height)
		gp.set_color({20, 20, 20, 255})
		gp.clear()

		gp.push_transform()
		{
			gp.set_color(255)

			// Move to the right area of the viewport
			gp.translate(f32(h.x) * 0.5, f32(h.y))

			gp.set_image(0, image_rect)

			width  := gp.get_image_width(image_rect)
			height := gp.get_image_height(image_rect)
			size := gp.Vec2{f32(width), f32(height)}

			scale := size * 2

			gp.draw_textured_rect(0, {
				src = {{0, 0}, size},
				dst = {-scale/2, scale},
			})
		}
		gp.pop_transform();
	}
}

sample_rect_shutdown :: proc () {
	gp.destroy_image(image_rect)
}
