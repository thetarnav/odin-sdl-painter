// Math substrate: rect and matrix types plus converters.
// ----------------------------------------------------------------------------
// Internal transform storage is Mat (2x3, 6 floats). Mat3 exists only at the
// boundary for interop. Affine compose converts through Mat3 with the builtin
// * operator; per-vertex transform is a single builtin m * vec3.
package sdl_painter

import "core:math/linalg"

Vec2  :: [2]f32
Vec2i :: [2]i32

Point :: Vec2

Line :: struct {a, b: Point}

Triangle :: struct {a, b, c: Point}

Rect_Vec2 :: struct {
	using pos:  Vec2,
	      size: Vec2,
}

Rect_Vec2i :: struct {
	using pos:  Vec2i,
	      size: Vec2i,
}

Mat  :: matrix[2, 3]f32
Mat3 :: matrix[3, 3]f32

MAT_IDENTITY :: Mat{1, 0, 0, 0, 1, 0}

#assert(size_of(Mat) == 24, "Mat must stay 6 floats (2x3)")
#assert(size_of(Mat3) == 36, "Mat3 must stay 9 floats (3x3)")
#assert(size_of(Rect_Vec2) == 16, "Rect_Vec2 layout changed")
#assert(size_of(Rect_Vec2i) == 16, "Rect_Vec2i layout changed")

mat_identity :: proc() -> Mat {
	return MAT_IDENTITY
}

mat3_identity :: proc() -> Mat3 {
	return linalg.identity(Mat3)
}

// Lift an affine 2x3 matrix to homogeneous 3x3 (bottom row 0, 0, 1).
to_mat3 :: proc(m: Mat) -> Mat3 {
	return Mat3{m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], 0, 0, 1}
}

// Project a homogeneous 3x3 matrix back to affine 2x3 (drops bottom row).
from_mat3 :: proc(m: Mat3) -> Mat {
	return Mat{m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2]}
}

// Compose affine transforms: result applies b first, then a.
// Replaces _mul_projection_transform.
compose :: proc(a, b: Mat) -> Mat {
	return from_mat3(to_mat3(a) * to_mat3(b))
}

// Apply an affine transform to a point. Replaces _mat3_mul_vec2.
transform_point :: proc(m: Mat, p: Vec2) -> Vec2 {
	return m * [3]f32{p.x, p.y, 1}
}

// Convert an integer rect to a float rect (draw-group boundary helper).
rect_to_float :: proc(r: Rect_Vec2i) -> Rect_Vec2 {
	return Rect_Vec2{pos = {f32(r.pos.x), f32(r.pos.y)}, size = {f32(r.size.x), f32(r.size.y)}}
}
