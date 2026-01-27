#!/usr/bin/env python3
"""
Generate complete SHA256 compression function in SPIR-V assembly.
This fully unrolls all 64 rounds for maximum performance.
"""

K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

def main():
    lines = []

    # Header
    lines.append("""; SHA256 compression function - Complete SPIR-V implementation (generated)
; Export: sha256_compress([8]u32 state, [16]u32 block) -> [8]u32
;
; SPIR-V
; Version: 1.0

               OpCapability Shader
               OpCapability Linkage
               OpMemoryModel Logical GLSL450

               OpDecorate %sha256_compress LinkageAttributes "sha256_compress" Export

; Types
       %void = OpTypeVoid
        %u32 = OpTypeInt 32 0
     %u32_8  = OpConstant %u32 8
    %u32_16  = OpConstant %u32 16
  %arr_8_u32 = OpTypeArray %u32 %u32_8
 %arr_16_u32 = OpTypeArray %u32 %u32_16
    %fn_type = OpTypeFunction %arr_8_u32 %arr_8_u32 %arr_16_u32

; Constants
      %c0  = OpConstant %u32 0
      %c2  = OpConstant %u32 2
      %c3  = OpConstant %u32 3
      %c6  = OpConstant %u32 6
      %c7  = OpConstant %u32 7
      %c9  = OpConstant %u32 9
     %c10  = OpConstant %u32 10
     %c11  = OpConstant %u32 11
     %c13  = OpConstant %u32 13
     %c14  = OpConstant %u32 14
     %c15  = OpConstant %u32 15
     %c17  = OpConstant %u32 17
     %c18  = OpConstant %u32 18
     %c19  = OpConstant %u32 19
     %c21  = OpConstant %u32 21
     %c22  = OpConstant %u32 22
     %c25  = OpConstant %u32 25
     %c26  = OpConstant %u32 26
     %c30  = OpConstant %u32 30
""")

    # Round constants
    for i, k in enumerate(K):
        lines.append(f"  %k{i} = OpConstant %u32 0x{k:08x}")
    lines.append("")

    # Function header
    lines.append("""; Main Function
%sha256_compress = OpFunction %arr_8_u32 None %fn_type
      %state = OpFunctionParameter %arr_8_u32
      %block = OpFunctionParameter %arr_16_u32
      %entry = OpLabel

; Extract initial hash values
       %h0 = OpCompositeExtract %u32 %state 0
       %h1 = OpCompositeExtract %u32 %state 1
       %h2 = OpCompositeExtract %u32 %state 2
       %h3 = OpCompositeExtract %u32 %state 3
       %h4 = OpCompositeExtract %u32 %state 4
       %h5 = OpCompositeExtract %u32 %state 5
       %h6 = OpCompositeExtract %u32 %state 6
       %h7 = OpCompositeExtract %u32 %state 7

; Extract message block
""")

    for i in range(16):
        lines.append(f"       %w{i} = OpCompositeExtract %u32 %block {i}")

    lines.append("\n; Message Schedule W[16..63]")
    lines.append("; s0(x) = ROTR(x,7) ^ ROTR(x,18) ^ SHR(x,3)")
    lines.append("; s1(x) = ROTR(x,17) ^ ROTR(x,19) ^ SHR(x,10)")
    lines.append("; W[i] = W[i-16] + s0(W[i-15]) + W[i-7] + s1(W[i-2])")

    w = [f"%w{i}" for i in range(16)]  # Track W variable names

    for i in range(16, 64):
        # W[i] = W[i-16] + s0(W[i-15]) + W[i-7] + s1(W[i-2])
        w_16 = w[i-16]
        w_15 = w[i-15]
        w_7 = w[i-7]
        w_2 = w[i-2]
        p = f"w{i}_"

        lines.append(f"\n; W[{i}] = W[{i-16}] + s0(W[{i-15}]) + W[{i-7}] + s1(W[{i-2}])")
        # s0(w_15) = ROTR(w_15,7) ^ ROTR(w_15,18) ^ SHR(w_15,3)
        lines.append(f"    %{p}r7 = OpShiftRightLogical %u32 {w_15} %c7")
        lines.append(f"    %{p}l25 = OpShiftLeftLogical %u32 {w_15} %c25")
        lines.append(f"    %{p}rot7 = OpBitwiseOr %u32 %{p}r7 %{p}l25")
        lines.append(f"    %{p}r18 = OpShiftRightLogical %u32 {w_15} %c18")
        lines.append(f"    %{p}l14 = OpShiftLeftLogical %u32 {w_15} %c14")
        lines.append(f"    %{p}rot18 = OpBitwiseOr %u32 %{p}r18 %{p}l14")
        lines.append(f"    %{p}shr3 = OpShiftRightLogical %u32 {w_15} %c3")
        lines.append(f"    %{p}x0 = OpBitwiseXor %u32 %{p}rot7 %{p}rot18")
        lines.append(f"    %{p}s0 = OpBitwiseXor %u32 %{p}x0 %{p}shr3")

        # s1(w_2) = ROTR(w_2,17) ^ ROTR(w_2,19) ^ SHR(w_2,10)
        lines.append(f"    %{p}r17 = OpShiftRightLogical %u32 {w_2} %c17")
        lines.append(f"    %{p}l15 = OpShiftLeftLogical %u32 {w_2} %c15")
        lines.append(f"    %{p}rot17 = OpBitwiseOr %u32 %{p}r17 %{p}l15")
        lines.append(f"    %{p}r19 = OpShiftRightLogical %u32 {w_2} %c19")
        lines.append(f"    %{p}l13 = OpShiftLeftLogical %u32 {w_2} %c13")
        lines.append(f"    %{p}rot19 = OpBitwiseOr %u32 %{p}r19 %{p}l13")
        lines.append(f"    %{p}shr10 = OpShiftRightLogical %u32 {w_2} %c10")
        lines.append(f"    %{p}x1 = OpBitwiseXor %u32 %{p}rot17 %{p}rot19")
        lines.append(f"    %{p}s1 = OpBitwiseXor %u32 %{p}x1 %{p}shr10")

        # W[i] = W[i-16] + s0 + W[i-7] + s1
        lines.append(f"    %{p}t0 = OpIAdd %u32 {w_16} %{p}s0")
        lines.append(f"    %{p}t1 = OpIAdd %u32 %{p}t0 {w_7}")
        lines.append(f"    %w{i} = OpIAdd %u32 %{p}t1 %{p}s1")
        w.append(f"%w{i}")

    # Compression rounds
    lines.append("\n; =========================================================")
    lines.append("; Compression Rounds")
    lines.append("; S1(e) = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25)")
    lines.append("; Ch(e,f,g) = (e & f) ^ (~e & g)")
    lines.append("; T1 = h + S1 + Ch + K[i] + W[i]")
    lines.append("; S0(a) = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22)")
    lines.append("; Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)")
    lines.append("; T2 = S0 + Maj")
    lines.append("; h=g, g=f, f=e, e=d+T1, d=c, c=b, b=a, a=T1+T2")
    lines.append("; =========================================================")

    # Working variables
    a, b, c, d, e, f, g, h = "%h0", "%h1", "%h2", "%h3", "%h4", "%h5", "%h6", "%h7"

    for i in range(64):
        p = f"r{i}_"
        ki = f"%k{i}"
        wi = w[i]

        lines.append(f"\n; Round {i}")

        # S1(e) = ROTR(e,6) ^ ROTR(e,11) ^ ROTR(e,25)
        lines.append(f"    %{p}er6 = OpShiftRightLogical %u32 {e} %c6")
        lines.append(f"    %{p}el26 = OpShiftLeftLogical %u32 {e} %c26")
        lines.append(f"    %{p}erot6 = OpBitwiseOr %u32 %{p}er6 %{p}el26")
        lines.append(f"    %{p}er11 = OpShiftRightLogical %u32 {e} %c11")
        lines.append(f"    %{p}el21 = OpShiftLeftLogical %u32 {e} %c21")
        lines.append(f"    %{p}erot11 = OpBitwiseOr %u32 %{p}er11 %{p}el21")
        lines.append(f"    %{p}er25 = OpShiftRightLogical %u32 {e} %c25")
        lines.append(f"    %{p}el7 = OpShiftLeftLogical %u32 {e} %c7")
        lines.append(f"    %{p}erot25 = OpBitwiseOr %u32 %{p}er25 %{p}el7")
        lines.append(f"    %{p}S1t = OpBitwiseXor %u32 %{p}erot6 %{p}erot11")
        lines.append(f"    %{p}S1 = OpBitwiseXor %u32 %{p}S1t %{p}erot25")

        # Ch(e,f,g) = (e & f) ^ (~e & g)
        lines.append(f"    %{p}ef = OpBitwiseAnd %u32 {e} {f}")
        lines.append(f"    %{p}ne = OpNot %u32 {e}")
        lines.append(f"    %{p}neg = OpBitwiseAnd %u32 %{p}ne {g}")
        lines.append(f"    %{p}Ch = OpBitwiseXor %u32 %{p}ef %{p}neg")

        # T1 = h + S1 + Ch + K[i] + W[i]
        lines.append(f"    %{p}t1a = OpIAdd %u32 {h} %{p}S1")
        lines.append(f"    %{p}t1b = OpIAdd %u32 %{p}t1a %{p}Ch")
        lines.append(f"    %{p}t1c = OpIAdd %u32 %{p}t1b {ki}")
        lines.append(f"    %{p}T1 = OpIAdd %u32 %{p}t1c {wi}")

        # S0(a) = ROTR(a,2) ^ ROTR(a,13) ^ ROTR(a,22)
        lines.append(f"    %{p}ar2 = OpShiftRightLogical %u32 {a} %c2")
        lines.append(f"    %{p}al30 = OpShiftLeftLogical %u32 {a} %c30")
        lines.append(f"    %{p}arot2 = OpBitwiseOr %u32 %{p}ar2 %{p}al30")
        lines.append(f"    %{p}ar13 = OpShiftRightLogical %u32 {a} %c13")
        lines.append(f"    %{p}al19 = OpShiftLeftLogical %u32 {a} %c19")
        lines.append(f"    %{p}arot13 = OpBitwiseOr %u32 %{p}ar13 %{p}al19")
        lines.append(f"    %{p}ar22 = OpShiftRightLogical %u32 {a} %c22")
        lines.append(f"    %{p}al10 = OpShiftLeftLogical %u32 {a} %c10")
        lines.append(f"    %{p}arot22 = OpBitwiseOr %u32 %{p}ar22 %{p}al10")
        lines.append(f"    %{p}S0t = OpBitwiseXor %u32 %{p}arot2 %{p}arot13")
        lines.append(f"    %{p}S0 = OpBitwiseXor %u32 %{p}S0t %{p}arot22")

        # Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c)
        lines.append(f"    %{p}ab = OpBitwiseAnd %u32 {a} {b}")
        lines.append(f"    %{p}ac = OpBitwiseAnd %u32 {a} {c}")
        lines.append(f"    %{p}bc = OpBitwiseAnd %u32 {b} {c}")
        lines.append(f"    %{p}Majt = OpBitwiseXor %u32 %{p}ab %{p}ac")
        lines.append(f"    %{p}Maj = OpBitwiseXor %u32 %{p}Majt %{p}bc")

        # T2 = S0 + Maj
        lines.append(f"    %{p}T2 = OpIAdd %u32 %{p}S0 %{p}Maj")

        # New state: h=g, g=f, f=e, e=d+T1, d=c, c=b, b=a, a=T1+T2
        new_e = f"%{p}e_new"
        new_a = f"%{p}a_new"
        lines.append(f"    {new_e} = OpIAdd %u32 {d} %{p}T1")
        lines.append(f"    {new_a} = OpIAdd %u32 %{p}T1 %{p}T2")

        # Update working variables
        h = g
        g = f
        f = e
        e = new_e
        d = c
        c = b
        b = a
        a = new_a

    # Final additions
    lines.append("\n; Add compressed chunk to hash")
    lines.append(f"    %out0 = OpIAdd %u32 %h0 {a}")
    lines.append(f"    %out1 = OpIAdd %u32 %h1 {b}")
    lines.append(f"    %out2 = OpIAdd %u32 %h2 {c}")
    lines.append(f"    %out3 = OpIAdd %u32 %h3 {d}")
    lines.append(f"    %out4 = OpIAdd %u32 %h4 {e}")
    lines.append(f"    %out5 = OpIAdd %u32 %h5 {f}")
    lines.append(f"    %out6 = OpIAdd %u32 %h6 {g}")
    lines.append(f"    %out7 = OpIAdd %u32 %h7 {h}")

    lines.append("\n    %result = OpCompositeConstruct %arr_8_u32 %out0 %out1 %out2 %out3 %out4 %out5 %out6 %out7")
    lines.append("              OpReturnValue %result")
    lines.append("              OpFunctionEnd")

    print('\n'.join(lines))

if __name__ == "__main__":
    main()
