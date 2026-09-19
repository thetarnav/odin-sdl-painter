package sdl_painter

// ----------------------------------------------------------------------------
// Global Definitions and Utilities
// ----------------------------------------------------------------------------
// Sizing knobs are #config so embedders can tune them at build time:
//   odin build example/ -define:IMAGE_MAX=128
// Pinned by the shader interface — NOT configurable:
//   TEXTURE_SLOTS_MAX, INVALID_ID, IMPOSSIBLE_ID

PATH_MAX           :: #config(PATH_MAX, 512)
TEXTURE_SIZE_MAX   :: #config(TEXTURE_SIZE_MAX, 16 * 1024 * 1024) // 16 mb
IMAGE_MAX          :: #config(IMAGE_MAX, 64)
SHADER_MAX         :: #config(SHADER_MAX, 8)
PIPELINE_MAX       :: #config(PIPELINE_MAX, 16)
STATE_MAX          :: #config(STATE_MAX, 16)
TRANSFORMS_MAX     :: #config(TRANSFORMS_MAX, 64)
UNIFORM_FLOATS_MAX :: #config(UNIFORM_FLOATS_MAX, 8)
OPTIMIZER_DEPTH    :: #config(OPTIMIZER_DEPTH, 8)
VERTICES_MAX       :: #config(VERTICES_MAX, 65536)
COMMANDS_MAX       :: #config(COMMANDS_MAX, 16384)
MOVE_VERTICES_MAX  :: #config(MOVE_VERTICES_MAX, 96)

TEXTURE_SLOTS_MAX :: 4

INVALID_ID    :: 0
IMPOSSIBLE_ID :: 0xFFFFFFFF
