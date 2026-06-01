//! Initial-board sources for `viz simulate`. Built-in named patterns
//! (glider, blinker, beacon, pulsar, random), Conway-RLE files
//! (`.rle`), plaintext (`.cells`), and the existing flat-i32 `.bin`
//! convention used by the rest of viz.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};

/// Source for the initial board. CLI parsing in `main.rs` resolves the
/// `--pattern` argument into one of these variants.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Named built-in pattern (`glider`, `blinker`, `beacon`, `pulsar`,
    /// or `random`).
    Builtin(String),
    /// A pattern file — dispatched by extension (`.rle`, `.cells`,
    /// `.bin`).
    File(PathBuf),
}

/// Parse a `--pattern` CLI value: names without a path separator and
/// without an extension are built-ins; everything else is a file path.
/// A bare `random` resolves to `Builtin("random")`; a path
/// `./my-pattern.rle` resolves to `File(...)`.
pub fn parse(value: &str) -> Pattern {
    let trimmed = value.trim();
    let has_separator = trimmed.contains('/') || trimmed.contains('\\');
    let has_extension =
        std::path::Path::new(trimmed).extension().and_then(|e| e.to_str()).is_some_and(|e| !e.is_empty());
    if has_separator || has_extension {
        Pattern::File(PathBuf::from(trimmed))
    } else {
        Pattern::Builtin(trimmed.to_string())
    }
}

/// Build the initial board for `grid = (w, h)`. The result is a flat
/// row-major `Vec<i32>` of length `w * h`, suitable for direct upload
/// to a storage buffer.
pub fn build_initial_board(pattern: &Pattern, seed: Option<u64>, grid: (u32, u32)) -> Result<Vec<i32>> {
    let (w, h) = (grid.0 as i32, grid.1 as i32);
    if w <= 0 || h <= 0 {
        bail!("simulate grid must have positive dimensions, got {}x{}", w, h);
    }
    match pattern {
        Pattern::Builtin(name) => match name.as_str() {
            "glider" => Ok(centered(w, h, GLIDER, 3, 3)),
            "blinker" => Ok(centered(w, h, BLINKER, 3, 1)),
            "beacon" => Ok(centered(w, h, BEACON, 4, 4)),
            "pulsar" => Ok(centered(w, h, PULSAR, 13, 13)),
            "random" => Ok(random_board(
                seed.unwrap_or_else(time_seed),
                w,
                h,
                /* density = */ 0.30,
            )),
            other => Err(anyhow!(
                "unknown built-in pattern {:?}; known: glider, blinker, beacon, pulsar, random",
                other
            )),
        },
        Pattern::File(path) => {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            match ext.to_ascii_lowercase().as_str() {
                "rle" => load_rle(path, w, h),
                "cells" => load_cells(path, w, h),
                "bin" => load_bin(path, w, h),
                other => Err(anyhow!(
                    "unsupported pattern file extension {:?}; known: .rle, .cells, .bin",
                    other
                )),
            }
        }
    }
}

// =============================================================================
// Built-in patterns
// =============================================================================

/// `(x, y)` coordinate offsets of the alive cells, with `(0, 0)` at
/// the top-left of the pattern's bounding box.
type PatternCoords = &'static [(i32, i32)];

const GLIDER: PatternCoords = &[(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)];

const BLINKER: PatternCoords = &[(0, 0), (1, 0), (2, 0)];

const BEACON: PatternCoords = &[(0, 0), (1, 0), (0, 1), (3, 2), (2, 3), (3, 3)];

// 13×13 pulsar, period 3. Coordinates taken from Conwaylife.com.
const PULSAR: PatternCoords = &[
    (2, 0),
    (3, 0),
    (4, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (0, 2),
    (5, 2),
    (7, 2),
    (12, 2),
    (0, 3),
    (5, 3),
    (7, 3),
    (12, 3),
    (0, 4),
    (5, 4),
    (7, 4),
    (12, 4),
    (2, 5),
    (3, 5),
    (4, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (2, 7),
    (3, 7),
    (4, 7),
    (8, 7),
    (9, 7),
    (10, 7),
    (0, 8),
    (5, 8),
    (7, 8),
    (12, 8),
    (0, 9),
    (5, 9),
    (7, 9),
    (12, 9),
    (0, 10),
    (5, 10),
    (7, 10),
    (12, 10),
    (2, 12),
    (3, 12),
    (4, 12),
    (8, 12),
    (9, 12),
    (10, 12),
];

/// Place `coords` centered on a grid of `w x h`. Cells outside the grid
/// after centering are silently dropped.
fn centered(w: i32, h: i32, coords: PatternCoords, pw: i32, ph: i32) -> Vec<i32> {
    let mut board = vec![0i32; (w * h) as usize];
    let ox = (w - pw) / 2;
    let oy = (h - ph) / 2;
    for (x, y) in coords {
        let gx = ox + x;
        let gy = oy + y;
        if (0..w).contains(&gx) && (0..h).contains(&gy) {
            board[(gy * w + gx) as usize] = 1;
        }
    }
    board
}

/// xorshift64-derived random fill. `density` is the per-cell probability
/// of starting alive.
fn random_board(seed: u64, w: i32, h: i32, density: f32) -> Vec<i32> {
    let n = (w * h) as usize;
    let mut board = vec![0i32; n];
    let mut state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    let threshold = (density.clamp(0.0, 1.0) * (u32::MAX as f32)) as u32;
    for cell in board.iter_mut() {
        // xorshift64 step + take top 32 bits.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let r = (state >> 32) as u32;
        if r < threshold {
            *cell = 1;
        }
    }
    board
}

fn time_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15)
}

// =============================================================================
// File-format loaders
// =============================================================================

/// Conway Run-Length-Encoded format. Lines starting with `#` are
/// comments; one header line of the form
/// `x = W, y = H[, rule = B3/S23]` precedes the body, which is a
/// stream of `<count><tag>` tokens — `b`/`B` = dead, `o`/`O` = alive,
/// `$` = end of row, `!` = end of pattern. Counts default to 1.
/// Rules other than B3/S23 are accepted but not enforced — the SPIR-V
/// is the rule.
fn load_rle(path: &std::path::Path, w: i32, h: i32) -> Result<Vec<i32>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read RLE pattern file {:?}", path))?;

    let mut lines = text.lines().peekable();
    while let Some(line) = lines.peek() {
        if line.trim_start().starts_with('#') || line.trim().is_empty() {
            lines.next();
        } else {
            break;
        }
    }
    let header = lines.next().ok_or_else(|| anyhow!("RLE {:?}: missing header line", path))?;
    let (pw, ph) = parse_rle_header(header)
        .ok_or_else(|| anyhow!("RLE {:?}: cannot parse header {:?}", path, header))?;

    // The body is the rest of the file, concatenated, ignoring
    // whitespace and stopping at `!`.
    let body: String = lines.collect::<Vec<_>>().join("");
    let cells = parse_rle_body(&body, pw, ph)?;

    let mut board = vec![0i32; (w * h) as usize];
    let ox = (w - pw) / 2;
    let oy = (h - ph) / 2;
    for (x, y) in cells {
        let gx = ox + x;
        let gy = oy + y;
        if (0..w).contains(&gx) && (0..h).contains(&gy) {
            board[(gy * w + gx) as usize] = 1;
        }
    }
    Ok(board)
}

fn parse_rle_header(line: &str) -> Option<(i32, i32)> {
    // `x = 36, y = 9, rule = B3/S23`
    let mut x = None;
    let mut y = None;
    for piece in line.split(',') {
        let trimmed = piece.trim();
        // Each piece is `<name> = <value>`; split once on `=` and parse
        // the value as an integer. Pieces other than `x` and `y` (e.g.
        // `rule = B3/S23`) are silently skipped.
        let Some((name, value)) = trimmed.split_once('=') else {
            continue;
        };
        match name.trim() {
            "x" => x = value.trim().parse::<i32>().ok(),
            "y" => y = value.trim().parse::<i32>().ok(),
            _ => {}
        }
    }
    Some((x?, y?))
}

fn parse_rle_body(body: &str, pw: i32, ph: i32) -> Result<Vec<(i32, i32)>> {
    let mut cells = Vec::new();
    let mut x: i32 = 0;
    let mut y: i32 = 0;
    let mut count: i32 = 0;
    for ch in body.chars() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        if ch.is_ascii_digit() {
            count = count * 10 + (ch as i32 - '0' as i32);
            continue;
        }
        let run = if count == 0 { 1 } else { count };
        match ch {
            'b' | 'B' => {
                x += run;
            }
            'o' | 'O' => {
                for i in 0..run {
                    cells.push((x + i, y));
                }
                x += run;
            }
            '$' => {
                y += run;
                x = 0;
            }
            '!' => break,
            other => {
                bail!("RLE body contains unknown tag {:?}", other);
            }
        }
        count = 0;
        if x > pw + 1 || y > ph + 1 {
            // Generous bounds; some RLE files drift slightly beyond the
            // declared `x = `/`y = ` size due to trailing dead runs.
            // Hard-cap at the bounding box so malformed input can't
            // explode memory.
            break;
        }
    }
    Ok(cells)
}

/// Plaintext `.cells` format. `O` = alive, `.` = dead (or space),
/// `!`-prefixed lines are comments. Each non-comment line is one row.
fn load_cells(path: &std::path::Path, w: i32, h: i32) -> Result<Vec<i32>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("read .cells pattern file {:?}", path))?;
    let rows: Vec<&str> = text.lines().filter(|l| !l.trim_start().starts_with('!')).collect();
    let ph = rows.len() as i32;
    let pw = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0) as i32;

    let mut board = vec![0i32; (w * h) as usize];
    let ox = (w - pw) / 2;
    let oy = (h - ph) / 2;
    for (ry, row) in rows.iter().enumerate() {
        for (rx, ch) in row.chars().enumerate() {
            let alive = matches!(ch, 'O' | 'o' | '*' | '#');
            if !alive {
                continue;
            }
            let gx = ox + rx as i32;
            let gy = oy + ry as i32;
            if (0..w).contains(&gx) && (0..h).contains(&gy) {
                board[(gy * w + gx) as usize] = 1;
            }
        }
    }
    Ok(board)
}

/// Flat little-endian `i32` board, exactly `w * h * 4` bytes. Matches
/// the convention used by viz's `--storage-dir` `.bin` inputs.
fn load_bin(path: &std::path::Path, w: i32, h: i32) -> Result<Vec<i32>> {
    let bytes = std::fs::read(path).with_context(|| format!("read .bin pattern file {:?}", path))?;
    let expected = (w * h) as usize * 4;
    if bytes.len() != expected {
        bail!(
            "{:?}: expected {} bytes ({} cells * 4), got {}",
            path,
            expected,
            w * h,
            bytes.len()
        );
    }
    let board: Vec<i32> =
        bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    Ok(board)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pattern_classifies_builtins_vs_files() {
        assert!(matches!(parse("glider"), Pattern::Builtin(_)));
        assert!(matches!(parse("random"), Pattern::Builtin(_)));
        assert!(matches!(parse("foo.rle"), Pattern::File(_)));
        assert!(matches!(parse("./foo.cells"), Pattern::File(_)));
        assert!(matches!(parse("dir/foo"), Pattern::File(_)));
    }

    #[test]
    fn glider_lands_centered() {
        let board = build_initial_board(&Pattern::Builtin("glider".into()), Some(1), (8, 8)).unwrap();
        // The glider's bounding box is 3x3. Centered into 8x8, ox=2, oy=2.
        // Alive cells: (1,0), (2,1), (0,2), (1,2), (2,2) → absolute
        // (3,2), (4,3), (2,4), (3,4), (4,4).
        let alive: Vec<(i32, i32)> = (0..8)
            .flat_map(|y| (0..8).map(move |x| (x, y)))
            .filter(|(x, y)| board[(y * 8 + x) as usize] == 1)
            .collect();
        let mut expected = vec![(3, 2), (4, 3), (2, 4), (3, 4), (4, 4)];
        expected.sort();
        let mut actual = alive;
        actual.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn random_with_same_seed_is_deterministic() {
        let a = build_initial_board(&Pattern::Builtin("random".into()), Some(42), (16, 16)).unwrap();
        let b = build_initial_board(&Pattern::Builtin("random".into()), Some(42), (16, 16)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn rle_glider_round_trips() {
        let glider_rle = "x = 3, y = 3, rule = B3/S23\nbob$2bo$3o!\n";
        let dir = std::env::temp_dir().join("viz_simulate_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("glider.rle");
        std::fs::write(&path, glider_rle).unwrap();
        let board = build_initial_board(&Pattern::File(path), None, (8, 8)).unwrap();
        let mut alive: Vec<(i32, i32)> = (0..8)
            .flat_map(|y| (0..8).map(move |x| (x, y)))
            .filter(|(x, y)| board[(y * 8 + x) as usize] == 1)
            .collect();
        alive.sort();
        // RLE `bob$2bo$3o!` = (1,0), (2,1), (0,2), (1,2), (2,2).
        // 3x3 centered in 8x8 → ox=2, oy=2; absolute (3,2), (4,3),
        // (2,4), (3,4), (4,4).
        let mut expected = vec![(3, 2), (4, 3), (2, 4), (3, 4), (4, 4)];
        expected.sort();
        assert_eq!(alive, expected);
    }
}
