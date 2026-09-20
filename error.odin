// Error handling (Public)
// ----------------------------------------------------------------------------
package sdl_painter

import "core:log"

Error :: enum u32 {
	None,
	Setup_Image_Failed,
	Flush_Image_Failed,
	Create_Image_Failed,
	Create_Shader_Failed,
	Create_Pipeline_Failed,
	Create_Common_Shader_Failed,
	Create_White_Texture_Failed,
	Create_Transfer_Buffer_Failed,
	Create_Vertex_Buffer_Failed,
	Create_Common_Pipeline_Failed,
	Alloc_Failed,
	Uniforms_Full,
	Vertices_Full,
	Commands_Full,
	Flush_Failed,
	Acquire_Command_Buffer_Failed,
	Acquire_Swapchain_Texture_Failed,
}

// Error handling (Private)
// ----------------------------------------------------------------------------

_last_error: Error = .None

@(private)
_set_error :: proc (error: Error) {
	log.errorf("SDL_gp error: %s", get_error_message(error))
	_last_error = error
}

// Get the last error that occurred in SDL_gp. Returns ERROR_NONE if no
// error has occurred.
get_last_error :: proc () -> Error {
	return _last_error
}

// Get a human-readable string describing an Error value. Returns
// "Unknown error" if the error value is not recognized.
get_error_message :: proc (error: Error) -> string {
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
