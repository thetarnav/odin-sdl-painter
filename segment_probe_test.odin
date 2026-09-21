package sdl_painter

import "core:log"
import "core:sync"
import "core:testing"

_test_mutex: sync.Mutex

// Odin's test runner fails a test on any log.error; _set_error logs at error
// level, so tests that INTENTIONALLY trigger sticky errors swap in a quiet
// console logger first (must be assigned in the test body itself — context is
// passed by value, so a helper cannot do it for its caller). Sticky state is
// still asserted via get_last_error().

_test_reset :: proc (vertex_cap, cmd_cap: int) {
	_gp = {}
	_last_error = .None
	_gp.desc.max_vertices = vertex_cap
	_gp.desc.max_commands = cmd_cap
	_gp.vertices = make([dynamic]Vertex, 0, vertex_cap)
	_gp.commands = make([dynamic]_Command, 0, cmd_cap)
	_gp.uniforms = make([dynamic]Uniform, 0, cmd_cap)
	_gp.state.base_vertex  = 0
	_gp.state.base_uniform = 0
	_gp.state.base_command = 0
	_gp.state.viewport   = {0, {800, 600}}
	_gp.state.scissor    = {0, {-1, -1}}
	_gp.state.frame_size = {800, 600}
	_gp.frame_cmd_buffer = nil
	_gp.frame_texture    = nil
	_gp.segments_flushed = 0
}

_test_teardown :: proc () {
	delete(_gp.vertices)
	delete(_gp.commands)
	delete(_gp.uniforms)
	_gp = {}
	_last_error = .None
}

// Reserve + queue one draw with no custom pipeline (skips the uniform path).
_test_draw :: proc (pipeline: Pipeline, region: Region, count: int, primitive_type: Primitive_Type = .Triangles) -> bool {
	vertex_index := len(_gp.vertices)
	v := _next_vertices(count)
	if v == nil do return false
	for i in 0 ..< count do v[i] = {}
	return _queue_draw(pipeline, region, vertex_index, count, primitive_type)
}

@(test)
test_commands_full_visible :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(16, 4)
	defer _test_teardown()
	console := log.create_console_logger(.Fatal)
	defer log.destroy_console_logger(console)
	context.logger = console
	for _ in 0 ..< 4 {
		testing.expect(t, _next_command() != nil)
	}
	testing.expect(t, _next_command() == nil, "expected nil past capacity")
	testing.expect_value(t, get_last_error(), Error.Commands_Full)
}

@(test)
test_single_request_hard_error :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(8, 4)
	defer _test_teardown()
	console := log.create_console_logger(.Fatal)
	defer log.destroy_console_logger(console)
	context.logger = console
	testing.expect(t, _next_vertices(9) == nil, "oversize request must fail")
	testing.expect_value(t, get_last_error(), Error.Vertices_Full)
	// No flush was attempted (no handles stashed): generation untouched.
	testing.expect_value(t, _gp.segments_flushed, 0)
}

@(test)
test_uniforms_full_visible :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(16, 2)
	defer _test_teardown()
	console := log.create_console_logger(.Fatal)
	defer log.destroy_console_logger(console)
	context.logger = console
	testing.expect(t, _next_uniform() != nil)
	testing.expect(t, _next_uniform() != nil)
	testing.expect(t, _next_uniform() == nil, "expected nil past capacity")
	testing.expect_value(t, get_last_error(), Error.Uniforms_Full)
}

@(test)
test_triangles_merge_collapses :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(64, 16)
	defer _test_teardown()
	a := transmute(Pipeline)u32(0xA11CE)
	testing.expect(t, _test_draw(a, {{0.0, 0.0}, {0.1, 0.1}}, 3))
	testing.expect(t, _test_draw(a, {{0.2, 0.2}, {0.3, 0.3}}, 3))
	testing.expect_value(t, len(_gp.commands), 1)
	testing.expect_value(t, _gp.commands[0].args.draw.vertices_count, 6)
}

@(test)
test_overlap_blocks_backward_merge :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(64, 16)
	defer _test_teardown()
	s1 := transmute(Pipeline)u32(0x5EED1)
	s2 := transmute(Pipeline)u32(0x5EED2)
	testing.expect(t, _test_draw(s1, {{-0.5, -0.5}, {-0.3, -0.3}}, 3)) // cmd 0
	testing.expect(t, _test_draw(s2, {{-0.2, -0.2}, {0.2, 0.2}}, 3))   // cmd 1, other pipeline
	// Same state as cmd 0, disjoint from it, but overlapping cmd 1:
	// must merge FORWARD (new command), never backward into cmd 0.
	testing.expect(t, _test_draw(s1, {{0.0, 0.0}, {0.1, 0.1}}, 3))
	testing.expect_value(t, len(_gp.commands), 3)
	// Forward merge supersedes cmd 0: its vertices are duplicated into the new
	// tail command, so cmd 0 is skipped (.None), not left as Draw. (Legacy
	// behavior, unchanged by Batch 2 — verified against pre-change merger.)
	testing.expect_value(t, _gp.commands[0].cmd, _Command_Type.None)
	testing.expect_value(t, _gp.commands[0].args.draw.vertices_count, 3) // untouched
	fwd := _gp.commands[2].args.draw
	testing.expect_value(t, fwd.vertices_count, 6)
	testing.expect_value(t, fwd.vertex_index, 6)
}

@(test)
test_strips_never_merge :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(64, 16)
	defer _test_teardown()
	a := transmute(Pipeline)u32(0x5711)
	testing.expect(t, _test_draw(a, {{0.0, 0.0}, {0.1, 0.1}}, 4, .Triangle_Strip))
	testing.expect(t, _test_draw(a, {{0.2, 0.2}, {0.3, 0.3}}, 4, .Triangle_Strip))
	testing.expect_value(t, len(_gp.commands), 2)
	testing.expect(t, _test_draw(a, {{0.4, 0.4}, {0.5, 0.5}}, 2, .Line_Strip))
	testing.expect_value(t, len(_gp.commands), 3)
}

@(test)
test_inter_cmds_bounded :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	// 10 distinct pipelines, then a repeat of the oldest: the scan meets 9
	// non-matching draws before the match. Pre-fix this wrote inter_cmds[8]
	// (capacity 8) and trapped on bounds check; post-fix the scan stops at 8.
	_test_reset(256, 32)
	defer _test_teardown()
	for i in 1 ..< 11 {
		p := transmute(Pipeline)u32(0x1000 + i)
		lo := f32(i) * 0.01
		testing.expect(t, _test_draw(p, {{lo, lo}, {lo + 0.005, lo + 0.005}}, 3))
	}
	oldest := transmute(Pipeline)u32(0x1001)
	testing.expect(t, _test_draw(oldest, {{0.5, 0.5}, {0.6, 0.6}}, 3))
	testing.expect_value(t, len(_gp.commands), 11) // oldest repeat cannot reach back: new command
	testing.expect_value(t, get_last_error(), Error.None)
}

@(test)
test_segment_snapshot_resolves_scissor :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(16, 4)
	defer _test_teardown()
	_gp.state.viewport   = {{10, 20}, {400, 300}}
	_gp.state.scissor    = {{5, 5}, {100, 100}}
	_gp.state.frame_size = {800, 600}
	_snapshot_segment_state()
	testing.expect_value(t, _gp.seg_viewport, Recti{{10, 20}, {400, 300}})
	testing.expect_value(t, _gp.seg_scissor, Recti{{15, 25}, {100, 100}})
	_gp.state.scissor = {0, {-1, -1}}
	_snapshot_segment_state()
	testing.expect_value(t, _gp.seg_scissor, Recti{{0, 0}, {800, 600}})
}
