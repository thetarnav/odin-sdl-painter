package example

import "core:c"
import "core:math/rand"
import sdl "vendor:sdl3"
import gp ".."

MAX_IMAGES    :: 30
TRIGGER_FRAME :: 60

_images_setup: [MAX_IMAGES]gp.Image
_images_frame: [MAX_IMAGES]gp.Image

sample_load_images_setup :: proc () {

	surface := sdl.LoadSurface(#directory+"../images/hello-world.png")
	if surface == nil {
		sdl.Log("Failed to load image: %s", sdl.GetError())
	}

	for &i in _images_setup {
		i = gp.create_image(surface)
	}

	sdl.DestroySurface(surface)
}

frame_count: int

sample_load_images_render :: proc (delta_time_ms: u64) {

	sdl.Log("Frame count: %i", frame_count)

	window_width, window_height: c.int
	sdl.GetWindowSize(_context.window, &window_width, &window_height)
	window := [2]int{int(window_width), int(window_height)}

	gp.set_blend_mode(.Blend)

	// trigger loading images after a certain number of frames
	if frame_count == TRIGGER_FRAME {

		surface := sdl.LoadSurface(#directory+"../images/sprites.png")
		if surface == nil {
			sdl.Log("Failed to load image: %s", sdl.GetError())
		}

		for &i in _images_frame {
			i = gp.create_image(surface)
		}

		sdl.DestroySurface(surface)
	}

	// Render the images loaded during the frame that triggered the loading
	// and onwards
	if frame_count >= TRIGGER_FRAME {

		@static
		tile_region := [3]gp.Rect_Vec2{
			{{0, 0}, {32, 32}}, // tile 1
			{{32, 0}, {32, 32}}, // tile 2
			{{64, 0}, {32, 32}}, // tile 3
		}

		gp.set_color({255, 255, 255, 255})

		// randomly select one of the loaded images to render
		gp.set_image(0, _images_frame[rand.int_max(MAX_IMAGES)])

		for i in 0..<4096 {
			x := rand.int_max(window.x)
			y := rand.int_max(window.y)

			src_rect := tile_region[i % 3]
			dst_rect := gp.Rect_Vec2{{f32(x), f32(y)}, {64, 64}}

			gp.draw_textured_rect(0, {dst_rect, src_rect})
		}

		gp.reset_image(0)
	}

	// Render the images loaded during setup all the time
	image := _images_setup[sdl.rand(MAX_IMAGES)]

	gp.set_image(0, image)

	gp.set_color({255, 255, 255, 255})

	width  := gp.get_image_width(image)
	height := gp.get_image_height(image)
	size   := gp.Vec2{f32(width), f32(height)}

	src_rect := gp.Rect_Vec2{{0, 0}, {f32(width), f32(height)}}
	dst_rect := gp.Rect_Vec2{(gp.Vec2(window) - size) * 0.5, size}

	gp.draw_textured_rect(0, {dst_rect, src_rect})
	gp.reset_image(0)

	gp.reset_blend_mode()

	frame_count += 1
}

sample_load_images_shutdown :: proc () {
	for i in _images_setup do gp.destroy_image(i)
	for i in _images_frame do gp.destroy_image(i)
}
