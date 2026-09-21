package example

import "core:c"
import "core:log"
import sdl "vendor:sdl3"
import gp ".."

image_rect: gp.Image

sample_rect_setup :: proc () {
	surface := sdl.LoadSurface(#directory+"../images/hello-world.png")
	if surface == nil {
		log.errorf("Failed to load image: %s", sdl.GetError())
	}

	image_rect = gp.make_image_from_surface(surface) or_else panic(gp.get_error_message(gp.get_last_error()))

	sdl.DestroySurface(surface)
}

sample_rect_render :: proc (delta_time_ms: u64) {

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window := [2]int{int(window_width), int(window_height)}

	h := window/2

	// Draw a red filled rectangle.
	{
		gp.set_viewport(0, 0, h.x, window.y)
		gp.set_color({10, 10, 10, 255})
		gp.clear()

		if gp.transform_scope() {
			gp.set_color({255, 0, 0, 255})

			// Move to the left area of the viewport
			gp.translate(Vec2(h) * {0.5, 1})

			half_shape := f32(window.x) * 0.15 // 15% of the viewport width

			gp.draw_rect(-half_shape, half_shape * 2)
		}
	}

	// Draw a textured rectangle keeping it's original color.
	{
		gp.set_viewport(h.x, 0, h.x, window.y)
		gp.set_color({20, 20, 20, 255})
		gp.clear()

		gp.push_transform()
		{
			size := Vec2(gp.get_image_size(image_rect))

			gp.set_color(255)

			// Move to the right area of the viewport
			gp.translate(Vec2(h) * {0.5, 1})

			gp.set_image(0, image_rect)
			gp.draw_textured_rect(0,
				src = {{0, 0}, size},
				dst = {-size, size*2},
			)
		}
		gp.pop_transform();
	}
}

sample_rect_shutdown :: proc () {
	gp.destroy_image(image_rect)
}
