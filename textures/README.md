# Textures

Shadertoy media-library images used by playground ports. Filenames are
Shadertoy's content hashes (from each shader's descriptor JSON).

## Snail (`testfiles/playground/snail.wyn`, Shadertoy ld3Gz2)

| Channel  | File                                                                   | Role |
|----------|------------------------------------------------------------------------|------|
| iChannel1 | `ad56fba948dfba9ae698198c109e71f118a54d209c0ea50d77ea546abad89c57.png` | 512² grayscale noise — surface displacement / sparkle (`text1`) |
| iChannel2 | `92d7758c402f0927011ca8d0a7e40251439fba3a1dac26f5b8b62026323501aa.jpg` | 1024² organic — shell bump, body, plant/leaf colour, background blur |
| iChannel3 | `f735bee5b64ef98879dc618b016ecf7939a5756040c2cde21ccb15e69a6e1cfb.png` | 256² RGBA normal map — leaf bumps |

Run (binding name : file):

```
viz run snail.spv \
  --texture iChannel1:textures/ad56fba948dfba9ae698198c109e71f118a54d209c0ea50d77ea546abad89c57.png \
  --texture iChannel2:textures/92d7758c402f0927011ca8d0a7e40251439fba3a1dac26f5b8b62026323501aa.jpg \
  --texture iChannel3:textures/f735bee5b64ef98879dc618b016ecf7939a5756040c2cde21ccb15e69a6e1cfb.png
```

Source: <https://www.shadertoy.com/media/a/&lt;hash&gt;>. These are
Shadertoy's stock textures; redistribution terms are theirs.
