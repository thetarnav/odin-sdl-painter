# odin-sdl-painter

Odin immediate-mode 2D painter over SDL3 GPU (`vendor:sdl3`).

Adapted from [SDL_gp](https://github.com/n67094/SDL_gp).

## Features

- Batched rect / textured-rect / line / triangle / point drawing
- Transform stack (`push_transform`, `translate`, `rotate`, `scale`)
- Custom pipelines with uniforms, blend modes, viewports, scissors
- Image management with mid-frame creation support

## Usage

```odin
import gp "path/to/sdl_painter"

// once
gp.setup({window = window, gpu_device = device})

// per frame
swapchain_texture := /* acquire */
gp.begin({width, height}, cmd_buffer, swapchain_texture)
gp.set_color({255, 0, 0, 255})
gp.draw_rect({10, 10}, {50, 50})
gp.flush(cmd_buffer, swapchain_texture)
gp.end()

// at exit
gp.shutdown()
```

## License

MIT — Copyright (c) 2026 Damian Tarnawski, portions by nsix. See `LICENSE.txt`.
