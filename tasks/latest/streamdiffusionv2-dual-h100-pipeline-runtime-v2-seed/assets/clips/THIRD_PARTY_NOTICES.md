# Third-party media notices

The six `.npy` files in this directory are cropped/resized RGB excerpts made from the
official Blender Foundation 1080p film downloads below. They are redistributed under
Creative Commons Attribution 3.0 Unported (CC BY 3.0):
<https://creativecommons.org/licenses/by/3.0/>. No upstream
`StreamDiffusionV2/examples/*.mp4` media is used.

## Big Buck Bunny

- Title: *Big Buck Bunny* (2008)
- Attribution: © 2008 Blender Foundation / <https://www.bigbuckbunny.org/>
- Official source: <https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4.zip>
- License: CC BY 3.0
- Official license/attribution statement: <https://peach.blender.org/about/>
- `bbb_480p_129.npy`: source frames 3600–3728 inclusive at 30 fps.
- `bbb_square_257_cut.npy`: source frames 6300–6427 followed by 12900–13028; hard cut at output frame 128.

## Sintel

- Title: *Sintel* (2010)
- Attribution: © Blender Foundation / <https://durian.blender.org/>
- Official source: <https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv>
- License: CC BY 3.0
- Official license/attribution statement: <https://durian.blender.org/sharing/>
- `sintel_480p_129.npy`: source frames 2640–2768 inclusive at 24 fps.
- `sintel_square_513_cut.npy`: source frames 5520–5775 followed by 14640–14896; hard cut at output frame 256.

## Tears of Steel

- Title: *Tears of Steel* (2012)
- Attribution: © Blender Foundation / <https://mango.blender.org/>
- Official source: <https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip>
- License: CC BY 3.0
- Official license statement: <https://mango.blender.org/about/>
- `tears_square576_257.npy`: source frames 2400–2656 inclusive at 24 fps.
- `tears_portrait_513_cut.npy`: source frames 7440–7695 followed by 12960–13216; hard cut at output frame 256.

The frame ranges above use zero-based source frame numbers derived from the stated
constant source frame rate. `ffmpeg -ss` seeks and then decodes the exact requested
number of frames; `clips_manifest.json` records resulting shapes, hashes, and motion
statistics.
