package sdl_painter

// Noun-verb aliases (Public)
// ----------------------------------------------------------------------------
// Explicit setter/getter/reset form for every short-verb state procedure.
// The set_-prefixed form is canonical; these are thin procedure-value aliases
// so existing call sites keep compiling. Groups (viewport_set, scissor_set)
// cover all arities of the underlying overload set.

matrix_set       :: set_matrix
matrix_get       :: get_matrix
mat3_set         :: set_mat3
mat3_get         :: get_mat3
projection_set   :: set_projection
projection_reset :: reset_projection
transform_push   :: push_transform
transform_pop    :: pop_transform
transform_reset  :: reset_transform
pipeline_set     :: set_pipeline
pipeline_reset   :: reset_pipeline
uniform_set      :: set_uniform
uniform_reset    :: reset_uniform
blend_mode_set   :: set_blend_mode
blend_mode_reset :: reset_blend_mode
color_set        :: set_color
color_get        :: get_color
color_reset      :: reset_color
image_set        :: set_image
image_reset      :: reset_image
sampler_set      :: set_sampler
sampler_reset    :: reset_sampler
viewport_set     :: set_viewport
viewport_reset   :: reset_viewport
scissor_set      :: set_scissor
scissor_reset    :: reset_scissor
state_reset      :: reset_state
