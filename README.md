# odin-sdl-painter

Idiomatic Odin immediate-mode 2D painter over SDL3 GPU (`vendor:sdl3`, no C toolchain).

## Features

- Batched rect / textured-rect / line / triangle / point drawing
- Transform stack (`push_transform`, `translate`, `rotate`, `scale`) with 2x3 matrix core
- Custom pipelines with uniforms, blend modes, viewports, scissors
- Image management with mid-frame creation support

## Requirements

- Odin `dev-2026-09` or newer
- SDL3 (via Odin's `vendor:sdl3`)

## Quickstart

```odin
import gp "path/to/sdl_painter"

// once
gp.setup(&gp.Desc{window = window, gpu_device = device})

// per frame
gp.begin(width, height)
gp.set_color({255, 0, 0, 255})
gp.draw_rect({{10, 10}, {50, 50}})
swapchain_texture := /* acquire */
gp.flush(cmd_buffer, swapchain_texture)
gp.end()

// at exit
gp.shutdown()
```

## Building the example

```sh
odin build example/ -out:/tmp/painter-example
odin run example/   # needs a display; arrow keys switch samples
```

## API overview

| Area | Procs |
|---|---|
| Frame | `setup`, `begin`, `flush`, `end`, `shutdown` |
| Draw | `draw_rect`, `draw_textured_rect`, `draw_line`, `draw_triangle`, `draw_point`, `draw` (+ batch plurals in each group) |
| State | `set_color`, `set_blend_mode`, `set_image`, `set_viewport`, `set_scissor`, `reset_state` (noun-verb aliases: `color_set`, `blend_mode_set`, `viewport_set`, … — colocated below their canonicals in painter.odin) |
| Transform | `push_transform`, `pop_transform`, `translate`, `rotate`, `scale`, `get_matrix`, `set_matrix` |
| | Scoped guard: `if gp.transform_scope() { ... }` |
| Rect variants | `set_viewport` / `set_scissor` take `(x, y, w, h)` or a `Recti` |

Sizing knobs (`IMAGE_MAX`, `VERTICES_MAX`, …) are `#config` — tune with `-define:IMAGE_MAX=128`.

## Allocators

`setup`, `shutdown`, `make_image`, and the internal `_image_flush` proc take an `allocator := context.allocator` parameter (`make_image` staging pixels are freed by `_image_flush`, so a destroy call must use the same allocator as its create call). Pipeline, shader, and image stores are `core:container/handle_map` static maps sized by the `_MAX` knobs — contextless setup/teardown with no allocator threading. Steady-state rendering stays allocation-free; all allocator traffic happens at setup/teardown.

## Shaders

`shaders/` ships opaque precompiled blobs (`.spv`/`.msl`/`.dxil`) selected at runtime by GPU backend; the `.glsl` sources are reference only and are never compiled.

## License

MIT — Copyright (c) 2026 Damian Tarnawski, portions by nsix. See `LICENSE.txt`.
