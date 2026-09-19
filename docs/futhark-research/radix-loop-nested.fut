-- Research fixture: stable binary radix pass, low 30 bits.
-- Kept large enough to expose function-boundary and inlining decisions.
def radix_bit [n] (xs: [n]u32) (bit: u32) : [n]u32 =
  let zero = map (\x -> if ((x >> bit) & 1u32) == 0u32 then 1i64 else 0i64) xs
  let prefix = scan (+) 0i64 zero
  let total = if n == 0 then 0i64 else prefix[n-1]
  let destinations = map3 (\i z p -> if z == 1 then p-1 else total+i-p) (iota n) zero prefix
  in scatter (replicate n 0u32) destinations xs

def radix_sort [n] (xs: [n]u32) : [n]u32 =
  loop xs for bit < 30 do radix_bit xs (u32.i64 bit)

entry main [m][n] (xss: [m][n]u32) = map radix_sort xss
