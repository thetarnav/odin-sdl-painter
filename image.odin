// Image (Public)
// ----------------------------------------------------------------------------
//
// An image is a wrapper around a GPU texture, with some additional metadata
// (width and height). The image creation function will create a GPU texture
// from an sdl.Surface and upload the surface pixels to the GPU texture.
//
// NOTE: The surface will be converted to the swapchain texture format if
// needed.
package sdl_painter

import sdl "vendor:sdl3"
import "core:c"
import "core:mem"

Sampler :: enum u32 {
	Point_Clamp  = 0,
	Point_Wrap   = 1,
	Linear_Clamp = 2,
	Linear_Wrap  = 3,
	Size         = 4,
}

Image :: struct {
	id: u32,
}

// Create an image from an sdl.Surface. Returns an invalid image if creation
// failed, use GetLastError() to get more information about the error.
create_image :: proc(surface: ^sdl.Surface) -> Image {
	assert(_img_ctx.initialized == _INIT_COOKIE)
	assert(_img_ctx.pool != nil)
	assert(_img_ctx.images_count < IMAGE_MAX, "Increase IMAGE_MAX to create more images")
	assert(surface != nil)

	inner_surface := surface

	texture_format := sdl.GetGPUSwapchainTextureFormat(_img_ctx.gpu_device, _img_ctx.window)
	pixel_format := sdl.GetPixelFormatFromGPUTextureFormat(texture_format)

	// Convert the surface to the swapchain texture format if needed

	converted := false
	if surface.format != pixel_format {
		sdl.LogWarn(
			c.int(sdl.LogCategory.APPLICATION),
			"Converting image pixel format from %s to %s",
			sdl.GetPixelFormatName(surface.format),
			sdl.GetPixelFormatName(pixel_format),
		)

		inner_surface = sdl.ConvertSurface(surface, pixel_format)

		if inner_surface == nil {
			_set_error(.Create_Image_Failed)
			return Image{id = INVALID_ID}
		}

		converted = true
	}
	defer if converted {
		sdl.DestroySurface(inner_surface)
	}

	texture_create_info := sdl.GPUTextureCreateInfo{
		type                 = .D2,
		format               = texture_format,
		usage                = {.SAMPLER},
		width                = u32(inner_surface.w),
		height               = u32(inner_surface.h),
		layer_count_or_depth = 1,
		num_levels           = 1,
		sample_count         = ._1,
	}

	texture := sdl.CreateGPUTexture(_img_ctx.gpu_device, texture_create_info)

	if texture == nil {
		_set_error(.Create_Image_Failed)
		return Image{id = INVALID_ID}
	}

	// Allocate image from resource

	slot := acquire_pool_slot(_img_ctx.pool)
	if slot == POOL_INVALID_SLOT {
		sdl.ReleaseGPUTexture(_img_ctx.gpu_device, texture)
		_set_error(.Create_Image_Failed)
		return Image{id = INVALID_ID}
	}

	_img_ctx.images[slot] = _Image{
		texture = texture,
		width   = u32(inner_surface.w),
		height  = u32(inner_surface.h),
	}

	// Create a pending image to be flushed later

	format_details := sdl.GetPixelFormatDetails(inner_surface.format)
	bpp := format_details.bytes_per_pixel
	size := uint(inner_surface.w) * uint(inner_surface.h) * uint(bpp)

	pixels_copy, alloc_err := mem.alloc(int(size))
	if alloc_err != .None || pixels_copy == nil {
		sdl.ReleaseGPUTexture(_img_ctx.gpu_device, texture)
		release_pool_slot(_img_ctx.pool, slot)
		_set_error(.Create_Image_Failed)
		return Image{id = INVALID_ID}
	}
	mem.copy(pixels_copy, inner_surface.pixels, int(size))

	_img_ctx.pending[_img_ctx.pending_count] = _Image_Pending{
		pixels = pixels_copy,
		width  = u32(inner_surface.w),
		height = u32(inner_surface.h),
		slot   = slot,
		bpp    = bpp,
	}
	_img_ctx.pending_count += 1

	// Destroy the converted surface if we created one

	_img_ctx.images_count += 1

	return Image{id = generate_pool_id(_img_ctx.pool, slot)}
}

// Destroy an image and free its resources.
destroy_image :: proc(image: Image) {
	assert(_img_ctx.initialized == _INIT_COOKIE)

	// TODO find a way to know if the image was already destroyed

	if image.id == INVALID_ID {
		return
	}

	slot := pool_id_to_slot(image.id)
	release_pool_slot(_img_ctx.pool, slot)

	inner_image := _img_ctx.images[slot]
	sdl.ReleaseGPUTexture(_img_ctx.gpu_device, inner_image.texture)

	_img_ctx.images[slot] = _Image{
		texture = nil,
		width   = 0,
		height  = 0,
	}
}

// Get the GPU texture associated with an image. Returns NULL if the image is
// invalid.
get_image_gpu_texture :: proc(image: Image) -> ^sdl.GPUTexture {
	assert(_img_ctx.initialized == _INIT_COOKIE)

	if image.id == INVALID_ID {
		return nil
	}

	slot := pool_id_to_slot(image.id)
	return _img_ctx.images[slot].texture
}

// Get the width of an image in pixels. Returns 0 if the image is invalid.
get_image_width :: proc(image: Image) -> i32 {
	assert(_img_ctx.initialized == _INIT_COOKIE)

	if image.id == INVALID_ID {
		return 0
	}

	slot := pool_id_to_slot(image.id)
	return i32(_img_ctx.images[slot].width)
}

// Get the height of an image in pixels. Returns 0 if the image is invalid.
get_image_height :: proc(image: Image) -> i32 {
	assert(_img_ctx.initialized == _INIT_COOKIE)

	if image.id == INVALID_ID {
		return 0
	}

	slot := pool_id_to_slot(image.id)
	return i32(_img_ctx.images[slot].height)
}

// Image (Private)
// ----------------------------------------------------------------------------

_Image :: struct {
	texture: ^sdl.GPUTexture,
	width:   u32,
	height:  u32,
}

_Image_Pending :: struct {
	pixels: rawptr,
	width:  u32,
	height: u32,
	slot:   i32,
	bpp:    u8,
}

_Image_Context :: struct {
	initialized:                 u32,
	images:                      []_Image,
	pending:                     [IMAGE_MAX + 1]_Image_Pending, // Images that are pending to be uploaded to the GPU + 1 for
		// the white texture for upload done during setup phase
	pending_count:               uint,
	pool:                        ^Pool,
	texture_transfer_buffer:     ^sdl.GPUTransferBuffer,
	texture_transfer_buffer_size: uint,
	gpu_device:                  ^sdl.GPUDevice,
	window:                      ^sdl.Window,
	images_count:                uint,
}

_img_ctx: _Image_Context

// Setup image resources management.
_image_setup :: proc(gpu_device: ^sdl.GPUDevice, window: ^sdl.Window) -> bool {
	assert(_img_ctx.initialized == 0)
	assert(gpu_device != nil)

	_img_ctx.initialized = _INIT_COOKIE

	_img_ctx.gpu_device = gpu_device
	_img_ctx.window = window

	// + 1 for the white image
	_img_ctx.pool = create_pool(IMAGE_MAX + 1)

	_img_ctx.images = make([]_Image, IMAGE_MAX + 1)

	transfer_buffer_create_info := sdl.GPUTransferBufferCreateInfo{
		usage = .UPLOAD,
		size  = TEXTURE_SIZE_MAX,
	}

	_img_ctx.texture_transfer_buffer = sdl.CreateGPUTransferBuffer(gpu_device, transfer_buffer_create_info)
	if _img_ctx.texture_transfer_buffer == nil {
		_set_error(.Setup_Image_Failed)
		return false
	}
	_img_ctx.texture_transfer_buffer_size = TEXTURE_SIZE_MAX

	return true
}

// Shutdown image resources management and free resources.
_image_shutdown :: proc() {
	assert(_img_ctx.initialized == _INIT_COOKIE)
	_img_ctx.initialized = 0

	destroy_pool(_img_ctx.pool)
	delete(_img_ctx.images)
	sdl.ReleaseGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer)
}

// Flush the image to the GPU. This will upload any pending images to the GPU.
// The texture transfer buffer will automatically be resized if needed.
// Returns false if an error occurred, use GetLastError() to get more
// information about the error.
_image_flush :: proc(cmd_buffer: ^sdl.GPUCommandBuffer) {
	if _img_ctx.pending_count == 0 {
		return
	}

	total_size: uint = 0
	for i: uint = 0; i < _img_ctx.pending_count; i += 1 {
		pending := &_img_ctx.pending[i]
		total_size += uint(pending.width) * uint(pending.height) * uint(pending.bpp)
	}

	// If the total size of pending images exceeds the transfer buffer size, we
	// need to cycle the transfer buffer to a larger size. We will double the size
	// of the transfer buffer until it is large enough to hold all pending images
	if total_size > _img_ctx.texture_transfer_buffer_size {
		new_size := _img_ctx.texture_transfer_buffer_size * 2

		if new_size == 0 {
			new_size = TEXTURE_SIZE_MAX
		}

		for new_size < total_size {
			new_size <<= 1
		}

		sdl.LogWarn(
			c.int(sdl.LogCategory.APPLICATION),
			"Total size of pending images (%zu) exceeds transfer buffer size (%zu), cycling transfer buffer",
			total_size,
			_img_ctx.texture_transfer_buffer_size,
		)

		sdl.ReleaseGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer)

		transfer_buffer_create_info := sdl.GPUTransferBufferCreateInfo{
			usage = .UPLOAD,
			size  = u32(new_size),
		}

		_img_ctx.texture_transfer_buffer = sdl.CreateGPUTransferBuffer(_img_ctx.gpu_device, transfer_buffer_create_info)
		if _img_ctx.texture_transfer_buffer == nil {
			_set_error(.Flush_Image_Failed)
			_img_ctx.texture_transfer_buffer_size = 0
			return
		}
		_img_ctx.texture_transfer_buffer_size = new_size
	}

	texture_transfer_ptr := sdl.MapGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer, false)

	copy_pass := sdl.BeginGPUCopyPass(cmd_buffer)
	offset: uint = 0

	for i: uint = 0; i < _img_ctx.pending_count; i += 1 {
		pending := &_img_ctx.pending[i]
		size := u32(pending.width) * u32(pending.height) * u32(pending.bpp)

		mem.copy(mem.ptr_offset(cast([^]u8)texture_transfer_ptr, int(offset)), pending.pixels, int(size))

		transfer_info := sdl.GPUTextureTransferInfo{
			transfer_buffer = _img_ctx.texture_transfer_buffer,
			offset          = u32(offset),
		}

		region := sdl.GPUTextureRegion{
			texture = _img_ctx.images[pending.slot].texture,
			w       = pending.width,
			h       = pending.height,
			d       = 1,
		}

		sdl.UploadToGPUTexture(copy_pass, transfer_info, region, false)

		offset += uint(size)

		mem.free(pending.pixels)
		pending.pixels = nil
	}

	sdl.EndGPUCopyPass(copy_pass)
	sdl.UnmapGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer)

	_img_ctx.pending_count = 0
}
