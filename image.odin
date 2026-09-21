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
import "base:builtin"
import "core:log"
import "core:mem"
import hm "core:container/handle_map"

Sampler :: enum u32 {
	Point_Clamp,
	Point_Wrap,
	Linear_Clamp,
	Linear_Wrap,
}

#assert(len(Sampler) == 4, "Sampler must stay four entries")

Image :: distinct hm.Handle32

// Create an image from an sdl.Surface. Returns an invalid image if creation
// failed, use GetLastError() to get more information about the error.
make_image :: proc (surface: ^sdl.Surface, allocator := context.allocator) -> Image {
	assert(_img_ctx.initialized)
	assert(surface != nil)

	inner_surface := surface

	texture_format := sdl.GetGPUSwapchainTextureFormat(_img_ctx.gpu_device, _img_ctx.window)
	pixel_format := sdl.GetPixelFormatFromGPUTextureFormat(texture_format)

	// Convert the surface to the swapchain texture format if needed

	converted := false
	if surface.format != pixel_format {
		log.warnf("Converting image pixel format from %s to %s",
			sdl.GetPixelFormatName(surface.format), sdl.GetPixelFormatName(pixel_format))

		inner_surface = sdl.ConvertSurface(surface, pixel_format)

		if inner_surface == nil {
			_set_error(.Create_Image_Failed)
			return {}
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
		return {}
	}

	// Insert image record into the map; exhaustion keeps the sticky-error contract.

	handle, ok := hm.add(&_img_ctx.images, _Image{
		texture = texture,
		size    = {inner_surface.w, inner_surface.h}
	})
	if !ok {
		sdl.ReleaseGPUTexture(_img_ctx.gpu_device, texture)
		_set_error(.Create_Image_Failed)
		return {}
	}

	// Create a pending image to be flushed later

	format_details := sdl.GetPixelFormatDetails(inner_surface.format)
	bpp := format_details.bytes_per_pixel
	size := int(inner_surface.w) * int(inner_surface.h) * int(bpp)

	pixels_copy, alloc_err := mem.alloc(size, allocator = allocator)
	if alloc_err != .None || pixels_copy == nil {
		sdl.ReleaseGPUTexture(_img_ctx.gpu_device, texture)
		hm.remove(&_img_ctx.images, handle)
		_set_error(.Create_Image_Failed)
		return {}
	}
	mem.copy(pixels_copy, inner_surface.pixels, size)

	assert(len(_img_ctx.pending) < cap(_img_ctx.pending), "Increase IMAGE_MAX to create more images")
	append(&_img_ctx.pending, _Image_Pending{
		pixels = pixels_copy,
		size   = {inner_surface.w, inner_surface.h},
		handle = handle,
		bpp    = bpp,
	})

	// Destroy the converted surface if we created one

	return handle
}

// Destroy an image and free its resources.
destroy_image :: proc (image: Image) {
	assert(_img_ctx.initialized)

	rec, ok := hm.get(&_img_ctx.images, image)
	if !ok do return // already destroyed or foreign id: safe no-op

	sdl.ReleaseGPUTexture(_img_ctx.gpu_device, rec.texture)
	hm.remove(&_img_ctx.images, image)
}

// Get the GPU texture associated with an image. Returns NULL if the image is
// invalid.
get_image_gpu_texture :: proc (image: Image) -> ^sdl.GPUTexture {
	assert(_img_ctx.initialized)

	rec, ok := hm.get(&_img_ctx.images, image)
	return rec.texture if ok else nil
}

// Get the width of an image in pixels. Returns 0 if the image is invalid.
get_image_width :: proc (image: Image) -> int {
	assert(_img_ctx.initialized)

	rec, ok := hm.get(&_img_ctx.images, image)
	return int(rec.x) if ok else 0
}

// Get the height of an image in pixels. Returns 0 if the image is invalid.
get_image_height :: proc (image: Image) -> int {
	assert(_img_ctx.initialized)

	rec, ok := hm.get(&_img_ctx.images, image)
	return int(rec.y) if ok else 0
}

// Get the size of an image in pixels as a vector. Returns {0, 0} if the image is invalid.
get_image_size :: proc (image: Image) -> Vec2i {
	assert(_img_ctx.initialized)

	rec, ok := hm.get(&_img_ctx.images, image)
	return rec.size if ok else 0
}

// Discoverability alias for the size query; canonical form is get_image_size.
image_size :: get_image_size

// Image (Private)
// ----------------------------------------------------------------------------

_Image :: struct {
	using handle: Image,
	texture:      ^sdl.GPUTexture,
	using size:   Vec2i,
}

_Image_Pending :: struct {
	using handle: Image,
	pixels:       rawptr,
	using size:   Vec2i,
	bpp:          u8,
}

_Image_Context :: struct {
	initialized:                 bool,
	images:                      hm.Static_Handle_Map(IMAGE_MAX + 1, _Image, Image), // +1 white slot
	pending:                     [dynamic; IMAGE_MAX + 1]_Image_Pending, // Images that are pending to be uploaded to the GPU + 1 for
		// the white texture for upload done during setup phase
	texture_transfer_buffer:     ^sdl.GPUTransferBuffer,
	texture_transfer_buffer_size: int,
	gpu_device:                  ^sdl.GPUDevice,
	window:                      ^sdl.Window,
}

_img_ctx: _Image_Context

// Setup image resources management.
@(private)
_image_setup :: proc (gpu_device: ^sdl.GPUDevice, window: ^sdl.Window) -> bool {
	assert(!_img_ctx.initialized)
	assert(gpu_device != nil)

	_img_ctx.initialized = true

	_img_ctx.gpu_device = gpu_device
	_img_ctx.window = window
	_img_ctx.images = {}

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
@(private)
_image_shutdown :: proc () {
	assert(_img_ctx.initialized)

	it := hm.iterator_make(&_img_ctx.images)
	for {
		rec, _, ok := hm.iterate(&it)
		if !ok do break
		sdl.ReleaseGPUTexture(_img_ctx.gpu_device, rec.texture)
	}

	sdl.ReleaseGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer)

	_img_ctx = {}
}

// Flush the image to the GPU. This will upload any pending images to the GPU.
// The texture transfer buffer will automatically be resized if needed.
// Returns false if an error occurred, use GetLastError() to get more
// information about the error.
@(private)
_image_flush :: proc (cmd_buffer: ^sdl.GPUCommandBuffer, allocator := context.allocator) {
	if len(_img_ctx.pending) == 0 {
		return
	}

	total_size: int
	for &pending in _img_ctx.pending {
		total_size += int(pending.x) * int(pending.y) * int(pending.bpp)
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

		log.warnf(
			"Total size of pending images (%v) exceeds transfer buffer size (%v), cycling transfer buffer",
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
	offset: int

	for &pending in _img_ctx.pending {
		size := int(pending.x) * int(pending.y) * int(pending.bpp)

		rec, ok := hm.get(&_img_ctx.images, pending.handle)
		if !ok {
			// Destroyed before flush: drop staging pixels, skip upload.
			mem.free(pending.pixels, allocator)
			pending.pixels = nil
			continue
		}

		mem.copy(mem.ptr_offset(cast([^]u8)texture_transfer_ptr, offset), pending.pixels, size)

		transfer_info := sdl.GPUTextureTransferInfo{
			transfer_buffer = _img_ctx.texture_transfer_buffer,
			offset          = u32(offset),
		}

		region := sdl.GPUTextureRegion{
			texture = rec.texture,
			w       = u32(pending.x),
			h       = u32(pending.y),
			d       = 1,
		}

		sdl.UploadToGPUTexture(copy_pass, transfer_info, region, false)

		offset += size

		mem.free(pending.pixels, allocator)
		pending.pixels = nil
	}

	sdl.EndGPUCopyPass(copy_pass)
	sdl.UnmapGPUTransferBuffer(_img_ctx.gpu_device, _img_ctx.texture_transfer_buffer)

	builtin.clear(&_img_ctx.pending)
}
