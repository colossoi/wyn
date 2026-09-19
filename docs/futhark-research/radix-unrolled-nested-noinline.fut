-- Research fixture: stable binary radix pass, low 30 bits.
-- Kept large enough to expose function-boundary and inlining decisions.
#[noinline]
def radix_bit [n] (xs: [n]u32) (bit: u32) : [n]u32 =
  let zero = map (\x -> if ((x >> bit) & 1u32) == 0u32 then 1i64 else 0i64) xs
  let prefix = scan (+) 0i64 zero
  let total = if n == 0 then 0i64 else prefix[n-1]
  let destinations = map3 (\i z p -> if z == 1 then p-1 else total+i-p) (iota n) zero prefix
  in scatter (replicate n 0u32) destinations xs

def radix_sort [n] (xs: [n]u32) : [n]u32 =
  let p1 = radix_bit xs 0u32
  let p2 = radix_bit p1 1u32
  let p3 = radix_bit p2 2u32
  let p4 = radix_bit p3 3u32
  let p5 = radix_bit p4 4u32
  let p6 = radix_bit p5 5u32
  let p7 = radix_bit p6 6u32
  let p8 = radix_bit p7 7u32
  let p9 = radix_bit p8 8u32
  let p10 = radix_bit p9 9u32
  let p11 = radix_bit p10 10u32
  let p12 = radix_bit p11 11u32
  let p13 = radix_bit p12 12u32
  let p14 = radix_bit p13 13u32
  let p15 = radix_bit p14 14u32
  let p16 = radix_bit p15 15u32
  let p17 = radix_bit p16 16u32
  let p18 = radix_bit p17 17u32
  let p19 = radix_bit p18 18u32
  let p20 = radix_bit p19 19u32
  let p21 = radix_bit p20 20u32
  let p22 = radix_bit p21 21u32
  let p23 = radix_bit p22 22u32
  let p24 = radix_bit p23 23u32
  let p25 = radix_bit p24 24u32
  let p26 = radix_bit p25 25u32
  let p27 = radix_bit p26 26u32
  let p28 = radix_bit p27 27u32
  let p29 = radix_bit p28 28u32
  let p30 = radix_bit p29 29u32
  in p30

entry main [m][n] (xss: [m][n]u32) = map radix_sort xss
