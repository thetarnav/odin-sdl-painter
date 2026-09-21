package example

import gp ".."

// Overflow torture: exceeds default VERTICES_MAX (65536) and COMMANDS_MAX
// (16384) every frame to exercise flush-and-retry segmentation, then covers
// all five primitive types. Passes iff get_last_error() stays .None and the
// screen shows a fully covered grid with no flicker or missing quads.
sample_overflow_setup :: proc () {}

sample_overflow_render :: proc (delta_time_ms: u64) {
	_ = delta_time_ms

	// 1. Vertex + command torrent: 20000 overlapping rects alternating blend
	// mode. Overlap blocks merging and the blend flip changes pipeline, so
	// (almost) every rect becomes its own command: ~20000 commands and
	// 120000 vertices, both past default caps.
	for i in 0 ..< 20000 {
		if i % 2 == 0 {
			gp.set_blend_mode(.None)
		} else {
			gp.set_blend_mode(.Blend)
		}
		gp.set_color({u8(i * 7), u8(i * 13), u8(255 - i), 255})
		gp.draw_rect(-0.9, -0.9, 1.8, 1.8)
	}
	gp.reset_blend_mode()

	// 2. Viewport/scissor churn across a segment boundary: state commands
	// must replay correctly in later segments (LOAD + restore path).
	gp.set_viewport(0, 0, WINDOW_WIDTH / 2, WINDOW_HEIGHT)
	gp.set_color({255, 255, 255, 255})
	gp.draw_rect(0, 0, 100, 100)
	gp.set_viewport(WINDOW_WIDTH / 2, 0, WINDOW_WIDTH / 2, WINDOW_HEIGHT)
	gp.set_scissor(10, 10, 200, 200)
	gp.set_color({0, 255, 0, 255})
	gp.draw_rect(10, 10, 200, 200)
	gp.reset_scissor()
	gp.reset_viewport()

	// 3. All five primitive types every frame (topology guard: strips must
	// render as strips, never merged).
	gp.set_color({255, 0, 0, 255})
	gp.draw_triangle(gp.Triangle{
		a = {-0.5, -0.5},
		b = {0.0, 0.5},
		c = {0.5, -0.5},
	})
	gp.draw_triangle_strip([]gp.Vec2{
		{0.6, -0.5},
		{0.7, 0.5},
		{0.8, -0.5},
		{0.9, 0.5},
	})
	gp.set_color({0, 0, 255, 255})
	gp.draw_line(gp.Vec2{-0.9, 0.9}, gp.Vec2{0.9, 0.9})
	gp.draw_line_strip([]gp.Vec2{
		{-0.9, 0.8},
		{0.9, 0.8},
		{0.9, 0.7},
	})
	gp.draw_point(gp.Point{0, 0})
}

sample_overflow_shutdown :: proc () {}
