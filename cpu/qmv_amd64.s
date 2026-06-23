//go:build amd64

#include "textflag.h"

// Per-lane right-shift counts: a u32 word holds 8 nibbles at bit offsets
// 0,4,...,28 — VPSRLVD shifts each of the 8 lanes by these to bring each nibble
// to bit 0, then VPAND keeps the low 4 bits. 8 lanes = one word per YMM.
DATA shr<>+0(SB)/4,  $0
DATA shr<>+4(SB)/4,  $4
DATA shr<>+8(SB)/4,  $8
DATA shr<>+12(SB)/4, $12
DATA shr<>+16(SB)/4, $16
DATA shr<>+20(SB)/4, $20
DATA shr<>+24(SB)/4, $24
DATA shr<>+28(SB)/4, $28
GLOBL shr<>(SB), RODATA|NOPTR, $32

DATA nmask<>+0(SB)/4,  $0x0000000F
DATA nmask<>+4(SB)/4,  $0x0000000F
DATA nmask<>+8(SB)/4,  $0x0000000F
DATA nmask<>+12(SB)/4, $0x0000000F
DATA nmask<>+16(SB)/4, $0x0000000F
DATA nmask<>+20(SB)/4, $0x0000000F
DATA nmask<>+24(SB)/4, $0x0000000F
DATA nmask<>+28(SB)/4, $0x0000000F
GLOBL nmask<>(SB), RODATA|NOPTR, $32

// func qChannel4AVX2(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) float32
TEXT ·qChannel4AVX2(SB), NOSPLIT, $0-52
	MOVQ w+0(FP), AX
	MOVQ x+8(FP), BX
	MOVQ scales+16(FP), CX
	MOVQ biases+24(FP), DX
	MOVQ groups+32(FP), SI
	MOVQ wordsPerGroup+40(FP), DI

	VMOVDQU shr<>(SB), Y10
	VMOVDQU nmask<>(SB), Y11
	VXORPS  Y14, Y14, Y14          // channel accumulator (8 lanes)

groupLoop:
	TESTQ SI, SI
	JE    done
	VXORPS Y0, Y0, Y0              // group sumXQ (8 lanes)
	VXORPS Y1, Y1, Y1              // group sumX  (8 lanes)
	MOVQ  DI, R8

wordLoop:
	TESTQ R8, R8
	JE    groupDone
	VPBROADCASTD (AX), Y2          // w in all 8 lanes
	ADDQ  $4, AX
	VPSRLVD Y10, Y2, Y2            // w >> {0,4,...,28}
	VPAND   Y11, Y2, Y2            // & 0xF  -> nibbles 0..7
	VCVTDQ2PS Y2, Y2              // uint32 -> float32 (q)
	VMOVUPS (BX), Y3              // x[0..7]
	ADDQ  $32, BX
	VFMADD231PS Y2, Y3, Y0        // sumXQ += q * x
	VADDPS  Y3, Y1, Y1            // sumX  += x
	DECQ  R8
	JMP   wordLoop

groupDone:
	VBROADCASTSS (CX), Y4          // scale[g] in 8 lanes
	ADDQ  $4, CX
	VBROADCASTSS (DX), Y5          // bias[g] in 8 lanes
	ADDQ  $4, DX
	VFMADD231PS Y0, Y4, Y14        // acc += scale * sumXQ
	VFMADD231PS Y1, Y5, Y14        // acc += bias  * sumX
	DECQ  SI
	JMP   groupLoop

done:
	// Horizontal sum of the 8 accumulator lanes (Y14) -> scalar.
	VEXTRACTF128 $1, Y14, X1       // X1 = high 4 lanes
	VADDPS  X1, X14, X14           // X14 = lo4 + hi4
	VHADDPS X14, X14, X14          // pairwise
	VHADDPS X14, X14, X14          // -> lane 0 = total
	MOVSS   X14, ret+48(FP)
	VZEROUPPER
	RET
