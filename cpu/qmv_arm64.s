//go:build arm64

#include "textflag.h"

// Per-lane right-shift counts for nibble extraction via USHL (negative = right).
// Lanes 0-3 → nibbles 0-3 (shifts 0,4,8,12); lanes 4-7 → nibbles 4-7 (16..28).
DATA shlo<>+0(SB)/4,  $0x00000000
DATA shlo<>+4(SB)/4,  $0xFFFFFFFC
DATA shlo<>+8(SB)/4,  $0xFFFFFFF8
DATA shlo<>+12(SB)/4, $0xFFFFFFF4
GLOBL shlo<>(SB), RODATA|NOPTR, $16

DATA shhi<>+0(SB)/4,  $0xFFFFFFF0
DATA shhi<>+4(SB)/4,  $0xFFFFFFEC
DATA shhi<>+8(SB)/4,  $0xFFFFFFE8
DATA shhi<>+12(SB)/4, $0xFFFFFFE4
GLOBL shhi<>(SB), RODATA|NOPTR, $16

DATA ones<>+0(SB)/4,  $0x3F800000
DATA ones<>+4(SB)/4,  $0x3F800000
DATA ones<>+8(SB)/4,  $0x3F800000
DATA ones<>+12(SB)/4, $0x3F800000
GLOBL ones<>(SB), RODATA|NOPTR, $16

DATA nmask<>+0(SB)/4,  $0x0000000F
DATA nmask<>+4(SB)/4,  $0x0000000F
DATA nmask<>+8(SB)/4,  $0x0000000F
DATA nmask<>+12(SB)/4, $0x0000000F
GLOBL nmask<>(SB), RODATA|NOPTR, $16

// func qChannel4NEON(w *uint32, x, scales, biases *float32, groups, wordsPerGroup int) float32
TEXT ·qChannel4NEON(SB), NOSPLIT, $32-52
	MOVD w+0(FP), R0
	MOVD x+8(FP), R1
	MOVD scales+16(FP), R2
	MOVD biases+24(FP), R3
	MOVD groups+32(FP), R4
	MOVD wordsPerGroup+40(FP), R5

	MOVD  $shlo<>(SB), R8
	VLD1  (R8), [V16.S4]
	MOVD  $shhi<>(SB), R8
	VLD1  (R8), [V17.S4]
	MOVD  $ones<>(SB), R8
	VLD1  (R8), [V19.S4]
	MOVD  $nmask<>(SB), R8
	VLD1  (R8), [V18.S4]        // 0x0000000F per 32-bit lane (low nibble mask)

	VEOR  V20.B16, V20.B16, V20.B16  // channel accumulator lo (4 lanes)
	VEOR  V21.B16, V21.B16, V21.B16  // channel accumulator hi (4 lanes)

groupLoop:
	CBZ   R4, done
	VEOR  V0.B16, V0.B16, V0.B16  // group sumXQ lo
	VEOR  V1.B16, V1.B16, V1.B16  // group sumXQ hi
	VEOR  V2.B16, V2.B16, V2.B16  // group sumX  lo
	VEOR  V3.B16, V3.B16, V3.B16  // group sumX  hi
	MOVD  R5, R6

wordLoop:
	CBZ   R6, groupDone
	MOVWU (R0), R7
	ADD   $4, R0
	VDUP  R7, V4.S4
	WORD  $0x6EB04485            // USHL V5.4S, V4.4S, V16.4S
	WORD  $0x6EB14486            // USHL V6.4S, V4.4S, V17.4S
	VAND  V18.B16, V5.B16, V5.B16
	VAND  V18.B16, V6.B16, V6.B16
	WORD  $0x6E21D8A5            // UCVTF V5.4S, V5.4S
	WORD  $0x6E21D8C6            // UCVTF V6.4S, V6.4S
	VLD1.P 32(R1), [V8.S4, V9.S4]
	VFMLA V5.S4, V8.S4, V0.S4    // sumXQlo += x_lo * q_lo
	VFMLA V6.S4, V9.S4, V1.S4    // sumXQhi += x_hi * q_hi
	VFMLA V19.S4, V8.S4, V2.S4   // sumXlo  += x_lo * 1
	VFMLA V19.S4, V9.S4, V3.S4
	SUB   $1, R6
	JMP   wordLoop

groupDone:
	// Fold this group's partial sums into the channel accumulator AS VECTORS
	// (no per-group scalar reduce): accLo += scale·sumXQlo + bias·sumXlo, etc.
	// scale/bias are broadcast across the 4 lanes via VDUP of their float bits.
	MOVWU (R2), R7
	ADD   $4, R2
	VDUP  R7, V23.S4            // scale[g] in all lanes
	MOVWU (R3), R7
	ADD   $4, R3
	VDUP  R7, V24.S4            // bias[g] in all lanes
	VFMLA V23.S4, V0.S4, V20.S4  // accLo += scale * sumXQlo
	VFMLA V24.S4, V2.S4, V20.S4  // accLo += bias  * sumXlo
	VFMLA V23.S4, V1.S4, V21.S4  // accHi += scale * sumXQhi
	VFMLA V24.S4, V3.S4, V21.S4  // accHi += bias  * sumXhi
	SUB   $1, R4
	JMP   groupLoop

done:
	// Reduce the 8 accumulator lanes (V20,V21) to one scalar — ONCE per channel.
	VST1  [V20.S4, V21.S4], (RSP)
	FMOVS 0(RSP), F20
	FMOVS 4(RSP), F11
	FADDS F11, F20, F20
	FMOVS 8(RSP), F11
	FADDS F11, F20, F20
	FMOVS 12(RSP), F11
	FADDS F11, F20, F20
	FMOVS 16(RSP), F11
	FADDS F11, F20, F20
	FMOVS 20(RSP), F11
	FADDS F11, F20, F20
	FMOVS 24(RSP), F11
	FADDS F11, F20, F20
	FMOVS 28(RSP), F11
	FADDS F11, F20, F20
	FMOVS F20, ret+48(FP)
	RET
