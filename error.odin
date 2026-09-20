// Error handling (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import sdl "vendor:sdl3"
import "core:c"

Error :: enum u32 {
	None                             = 0,
	Setup_Image_Failed               = 1,
	Flush_Image_Failed               = 2,
	Create_Image_Failed              = 3,
	Create_Shader_Failed             = 4,
	Create_Pipeline_Failed           = 5,
	Create_Common_Shader_Failed      = 6,
	Create_White_Texture_Failed      = 7,
	Create_Transfer_Buffer_Failed    = 8,
	Create_Vertex_Buffer_Failed      = 9,
	Create_Common_Pipeline_Failed    = 10,
	Alloc_Failed                     = 11,
	Uniforms_Full                    = 12,
	Vertices_Full                    = 13,
	Commands_Full                    = 14,
	Flush_Failed                     = 15,
	Acquire_Command_Buffer_Failed    = 16,
	Acquire_Swapchain_Texture_Failed = 17,
}

// Error handling (Private)
// ----------------------------------------------------------------------------

_last_error: Error = .None

@(private)
_set_error :: proc(error: Error) {
	sdl.LogError(c.int(sdl.LogCategory.VIDEO), "SDL_gp error: %s", get_error_message(error))
	_last_error = error
}

// Get the last error that occurred in SDL_gp. Returns ERROR_NONE if no
// error has occurred.
get_last_error :: proc() -> Error {
	return _last_error
}

// Get a human-readable string describing an Error value. Returns
// "Unknown error" if the error value is not recognized.
get_error_message :: proc(error: Error) -> cstring {
	switch error {
	case .None:                             return "No error"
	case .Setup_Image_Failed:               return "Failed to setup image resources"
	case .Flush_Image_Failed:               return "Failed to flush image resources"
	case .Create_Image_Failed:              return "Failed to create image"
	case .Create_Shader_Failed:             return "Failed to create shader"
	case .Create_Pipeline_Failed:           return "Failed to create pipeline"
	case .Create_Common_Shader_Failed:      return "Failed to create common shader"
	case .Create_White_Texture_Failed:      return "Failed to create white texture"
	case .Create_Vertex_Buffer_Failed:      return "Failed to create vertex buffer"
	case .Create_Transfer_Buffer_Failed:    return "Failed to create transfer buffer"
	case .Create_Common_Pipeline_Failed:    return "Failed to create common pipeline"
	case .Alloc_Failed:                     return "Failed to allocate memory"
	case .Uniforms_Full:                    return "Painter uniforms are full"
	case .Vertices_Full:                    return "Painter vertices are full"
	case .Commands_Full:                    return "Painter commands are full"
	case .Flush_Failed:                     return "Failed to flush painter"
	case .Acquire_Command_Buffer_Failed:    return "Failed to acquire GPU command buffer"
	case .Acquire_Swapchain_Texture_Failed: return "Failed to acquire swapchain texture"
	}
	return "Unknown error"
}
