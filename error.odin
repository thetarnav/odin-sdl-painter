// Error handling (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"
import "core:c"

Error :: enum u32 {
	NONE                             = 0,
	SETUP_IMAGE_FAILED               = 1,
	FLUSH_IMAGE_FAILED               = 2,
	CREATE_IMAGE_FAILED              = 3,
	CREATE_SHADER_FAILED             = 4,
	CREATE_PIPELINE_FAILED           = 5,
	CREATE_COMMON_SHADER_FAILED      = 6,
	CREATE_WHITE_TEXTURE_FAILED      = 7,
	CREATE_TRANSFER_BUFFER_FAILED    = 8,
	CREATE_VERTEX_BUFFER_FAILED      = 9,
	CREATE_COMMON_PIPELINE_FAILED    = 10,
	ALLOC_FAILED                     = 11,
	UNIFORMS_FULL                    = 12,
	VERTICES_FULL                    = 13,
	COMMANDS_FULL                    = 14,
	FLUSH_FAILED                     = 15,
	ACQUIRE_COMMAND_BUFFER_FAILED    = 16,
	ACQUIRE_SWAPCHAIN_TEXTURE_FAILED = 17,
}

// Error handling (Private)
// ----------------------------------------------------------------------------

@(private)
_last_error: Error = .NONE

@(private)
_set_error :: proc(error: Error) {
	sdl.LogError(c.int(sdl.LogCategory.VIDEO), "SDL_gp error: %s", GetErrorMessage(error))
	_last_error = error
}

// Get the last error that occurred in SDL_gp. Returns ERROR_NONE if no
// error has occurred.
GetLastError :: proc() -> Error {
	return _last_error
}

// Get a human-readable string describing an Error value. Returns
// "Unknown error" if the error value is not recognized.
GetErrorMessage :: proc(error: Error) -> cstring {
	switch error {
	case .NONE:                             return "No error"
	case .SETUP_IMAGE_FAILED:               return "Failed to setup image resources"
	case .FLUSH_IMAGE_FAILED:               return "Failed to flush image resources"
	case .CREATE_IMAGE_FAILED:              return "Failed to create image"
	case .CREATE_SHADER_FAILED:             return "Failed to create shader"
	case .CREATE_PIPELINE_FAILED:           return "Failed to create pipeline"
	case .CREATE_COMMON_SHADER_FAILED:      return "Failed to create common shader"
	case .CREATE_WHITE_TEXTURE_FAILED:      return "Failed to create white texture"
	case .CREATE_VERTEX_BUFFER_FAILED:      return "Failed to create vertex buffer"
	case .CREATE_TRANSFER_BUFFER_FAILED:    return "Failed to create transfer buffer"
	case .CREATE_COMMON_PIPELINE_FAILED:    return "Failed to create common pipeline"
	case .ALLOC_FAILED:                     return "Failed to allocate memory"
	case .UNIFORMS_FULL:                    return "Painter uniforms are full"
	case .VERTICES_FULL:                    return "Painter vertices are full"
	case .COMMANDS_FULL:                    return "Painter commands are full"
	case .FLUSH_FAILED:                     return "Failed to flush painter"
	case .ACQUIRE_COMMAND_BUFFER_FAILED:    return "Failed to acquire GPU command buffer"
	case .ACQUIRE_SWAPCHAIN_TEXTURE_FAILED: return "Failed to acquire swapchain texture"
	}
	return "Unknown error"
}
