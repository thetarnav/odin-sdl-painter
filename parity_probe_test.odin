package sdl_painter

import "core:sync"
import "core:testing"

// Deterministic script; golden values derived from the pre-change merger
// (prev-merge appends, next-merge duplicates tail, strips excluded).
_parity_run :: proc () {
	a := transmute(Pipeline)u32(0xA11CE)
	b := transmute(Pipeline)u32(0xB0B)
	_test_draw(a, {{-0.9, -0.9}, {-0.5, -0.5}}, 6) // cmd 0: rect-like
	_test_draw(a, {{-0.4, -0.9}, {0.0, -0.5}}, 6)  // merges into cmd 0 (prev, no inter)
	_test_draw(b, {{0.1, 0.1}, {0.5, 0.5}}, 2)     // cmd 1: other pipeline
	// Same pipe as cmd 0, disjoint from everything: prev-merge into cmd 0,
	// splicing cmd 1's 2 vertices forward.
	_test_draw(a, {{0.6, 0.6}, {0.9, 0.9}}, 3)
}

@(test)
test_parity_command_stream :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(64, 16)
	defer _test_teardown()
	_parity_run()
	testing.expect_value(t, len(_gp.commands), 2)
	testing.expect_value(t, len(_gp.vertices), 17) // 6+6+2+3, nothing lost
	testing.expect_value(t, _gp.commands[0].args.draw.vertices_count, 15)
	testing.expect_value(t, _gp.commands[0].args.draw.vertex_index, 0)
	testing.expect_value(t, _gp.commands[0].args.draw.region, Region{{-0.9, -0.9}, {0.9, 0.9}})
	testing.expect_value(t, _gp.commands[1].args.draw.vertices_count, 2)
	testing.expect_value(t, _gp.commands[1].args.draw.vertex_index, 15)
	testing.expect_value(t, get_last_error(), Error.None)
}

@(test)
test_all_five_primitive_types_queue :: proc (t: ^testing.T) {
	sync.mutex_lock(&_test_mutex)
	defer sync.mutex_unlock(&_test_mutex)
	_test_reset(64, 16)
	defer _test_teardown()
	types := [5]Primitive_Type{.Triangles, .Triangle_Strip, .Lines, .Line_Strip, .Points}
	for type, i in types {
		p := transmute(Pipeline)u32(0x2000 + i) // distinct pipeline per type
		testing.expect(t, _test_draw(p, {{-0.8, -0.8}, {0.8, 0.8}}, 3, type))
	}
	testing.expect_value(t, len(_gp.commands), 5)
	// Same pipeline, adjacent strips: still 2 commands (topology guard intact).
	_test_teardown()
	_test_reset(64, 16)
	s := transmute(Pipeline)u32(0x3000)
	testing.expect(t, _test_draw(s, {{-0.8, -0.8}, {-0.6, -0.6}}, 3, .Triangles))
	testing.expect(t, _test_draw(s, {{-0.5, -0.5}, {-0.3, -0.3}}, 3, .Triangles))
	testing.expect_value(t, len(_gp.commands), 1) // triangles DO merge on shared pipe
	testing.expect(t, _test_draw(s, {{-0.2, -0.2}, {0.0, 0.0}}, 3, .Triangle_Strip))
	testing.expect_value(t, len(_gp.commands), 2) // strip does NOT join them
}
