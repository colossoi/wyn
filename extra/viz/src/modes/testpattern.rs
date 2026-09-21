use crate::app::App;
use crate::modes::pipeline::{InteractiveOpts, RunSpec};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use wyn_host_interp::Program;

const TEST_PATTERN_SHADER: &str = r#"
// Resolution uniform (16-byte aligned)
struct Globals {
    resolution: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

// Vertex shader - fullscreen big triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

// Fragment shader - colored test pattern
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let res = globals.resolution.xy;
    let fragCoord = pos.xy;
    let uv = fragCoord / res;

    // Checkerboard using integer math (fragCoord >= 0, so truncation == floor)
    let grid_size = 64.0;
    let gx = i32(fragCoord.x / grid_size);
    let gy = i32(fragCoord.y / grid_size);
    let checker = f32((gx + gy) & 1);

    // Color gradient + diagonal stripes
    let r = uv.x;
    let g = uv.y;
    let b = checker * 0.5 + 0.5;
    let stripe = sin((uv.x + uv.y) * 20.0) * 0.5 + 0.5;

    return vec4<f32>(r * stripe, g, b * (1.0 - stripe * 0.3), 1.0);
}
"#;

const HOST: &str = r#"
(define-host-program :version 1)
(define-gpu-module 'shaders :format :wgsl :path "testpattern.wgsl")
(define-gpu-graphics 'pattern
  :vertex '(shaders "vs_main") :fragment '(shaders "fs_main")
  :parameters '((globals :buffer :read :layout (:size 16 :alignment 16 :fields ((source-resolution (:bytes 12) 0)))))
  :abi '((globals :uniform 0 0)) :vertex-inputs '() :color-outputs '((0 :caller))
  :depth-format nil :samples 1 :topology :triangle-list :front-face :counter-clockwise
  :cull :none :fill :fill :depth-test :disabled :depth-write nil :blend :replace :color-write t)
(define-host-entry 'pattern :function 'host-pattern :source-name "testpattern"
  :parameters '((globals :buffer :read :layout (:size 16 :alignment 16 :fields ((source-resolution (:bytes 12) 0))) :source-name "globals")
                (screen :texture :read-write :dimension :d2 :format :caller :samples 1 :source-name "screen"))
  :results '((screen :texture :read-write :dimension :d2 :format :caller :samples 1 :source-name "screen" :ownership :borrowed :alias screen)))
(defun host-pattern (globals screen)
  (let ((target (gpu-texture-view screen :usage :render-target :dimension :d2 :mip 0 :mip-count 1 :layer 0 :layer-count 1)))
    (gpu-draw 'pattern :args (list globals) :vertices nil :colors (list (list 0 target :load :store nil))
      :depth nil :viewport :target :scissor :target :draw '(:direct 3 1 0 0))
    screen))
"#;

pub fn run_test_pattern(max_frames: Option<u32>, verbose: bool) -> Result<()> {
    App::run(RunSpec {
        program: Program::parse(HOST)?,
        base: PathBuf::new(),
        sources: Some(BTreeMap::from([(
            "shaders".into(),
            TEST_PATTERN_SHADER.as_bytes().to_vec(),
        )])),
        inputs: HashMap::new(),
        outputs: HashMap::new(),
        constants: Vec::new(),
        dispatch: BTreeMap::new(),
        feedback: Vec::new(),
        opts: InteractiveOpts {
            max_frames,
            ..Default::default()
        },
        verbose,
    })
}
