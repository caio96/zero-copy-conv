	.text
	.file	"kernel-opt.c"
	.section	.rodata.cst32,"aM",@progbits,32
	.p2align	5, 0x0                          # -- Begin function conv_2d
.LCPI0_0:
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.short	255                             # 0xff
	.section	.rodata,"a",@progbits
	.p2align	1, 0x0
.LCPI0_1:
	.short	255                             # 0xff
	.text
	.globl	conv_2d
	.p2align	4, 0x90
	.type	conv_2d,@function
conv_2d:                                # @conv_2d
	.cfi_startproc
# %bb.0:                                # %entry
                                        # kill: def $r8d killed $r8d def $r8
                                        # kill: def $ecx killed $ecx def $rcx
	cmpl	$3, %ecx
	jl	.LBB0_7
# %bb.1:                                # %entry
	cmpl	$3, %r8d
	jl	.LBB0_7
# %bb.2:                                # %for.cond4.preheader.us.preheader
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	leal	-2(%r8), %eax
	addl	$-2, %ecx
	vmovdqu	(%rdx), %ymm2
	vmovdqu	48(%rdx), %ymm7
	vmovdqu	96(%rdx), %ymm9
	movl	%r8d, %r9d
	shll	$5, %r9d
	shll	$4, %r8d
	movq	%rax, %r10
	shlq	$4, %r10
	xorl	%r11d, %r11d
	vpunpckhbw	%ymm2, %ymm2, %ymm0     # ymm0 = ymm2[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpbroadcastw	.LCPI0_1(%rip), %ymm1   # ymm1 = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
	vpunpcklbw	%ymm2, %ymm2, %ymm2     # ymm2 = ymm2[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmovzxbw	32(%rdx), %ymm3         # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpxor	%xmm4, %xmm4, %xmm4
	vpunpckhbw	%ymm7, %ymm7, %ymm5     # ymm5 = ymm7[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmovzxbw	80(%rdx), %ymm6         # ymm6 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpunpcklbw	%ymm7, %ymm7, %ymm7     # ymm7 = ymm7[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpunpckhbw	%ymm9, %ymm9, %ymm8     # ymm8 = ymm9[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpunpcklbw	%ymm9, %ymm9, %ymm9     # ymm9 = ymm9[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmovzxbw	128(%rdx), %ymm10       # ymm10 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	movq	%r8, %rdx
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB0_3:                                # %for.cond4.preheader.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
	movq	%rsi, %r14
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB0_4:                                # %for.cond8.preheader.us
                                        #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	leal	(%r11,%r15), %ebp
	movslq	%ebp, %r12
	vmovdqu	(%rdi,%r12), %xmm11
	leal	(%r11,%r15), %ebp
	addl	$16, %ebp
	movslq	%ebp, %r12
	vinserti128	$1, (%rdi,%r12), %ymm11, %ymm11
	vpunpckhbw	%ymm11, %ymm11, %ymm12  # ymm12 = ymm11[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm0, %ymm12, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vpunpcklbw	%ymm11, %ymm11, %ymm11  # ymm11 = ymm11[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm2, %ymm11, %ymm11
	vpand	%ymm1, %ymm11, %ymm11
	vpackuswb	%ymm12, %ymm11, %ymm11
	leal	32(%r11,%r15), %ebp
	movslq	%ebp, %r12
	vpmovzxbw	(%rdi,%r12), %ymm12     # ymm12 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm3, %ymm12, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vextracti128	$1, %ymm12, %xmm13
	vpackuswb	%xmm13, %xmm12, %xmm12
	vextracti128	$1, %ymm11, %xmm14
	vpaddb	%xmm14, %xmm11, %xmm11
	vpshufd	$238, %xmm11, %xmm14            # xmm14 = xmm11[2,3,2,3]
	vpaddb	%xmm14, %xmm11, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %ebp
	vpackuswb	%xmm13, %xmm13, %xmm11
	vpaddb	%xmm11, %xmm12, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %r12d
	leal	(%rdx,%r15), %r13d
	movslq	%r13d, %r13
	vmovdqu	(%rdi,%r13), %xmm11
	leal	(%rdx,%r15), %r13d
	addl	$16, %r13d
	movslq	%r13d, %r13
	vinserti128	$1, (%rdi,%r13), %ymm11, %ymm11
	addb	%bpl, %r12b
	vpunpckhbw	%ymm11, %ymm11, %ymm12  # ymm12 = ymm11[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm5, %ymm12, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vpunpcklbw	%ymm11, %ymm11, %ymm11  # ymm11 = ymm11[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm7, %ymm11, %ymm11
	vpand	%ymm1, %ymm11, %ymm11
	vpackuswb	%ymm12, %ymm11, %ymm11
	leal	32(%rdx,%r15), %ebp
	movslq	%ebp, %r13
	vpmovzxbw	(%rdi,%r13), %ymm12     # ymm12 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm6, %ymm12, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vextracti128	$1, %ymm12, %xmm13
	vpackuswb	%xmm13, %xmm12, %xmm12
	vextracti128	$1, %ymm11, %xmm14
	vpaddb	%xmm14, %xmm11, %xmm11
	vpshufd	$238, %xmm11, %xmm14            # xmm14 = xmm11[2,3,2,3]
	vpaddb	%xmm14, %xmm11, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %r13d
	vpackuswb	%xmm13, %xmm13, %xmm11
	vpaddb	%xmm11, %xmm12, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %ebp
	addb	%r13b, %bpl
	addb	%r12b, %bpl
	leal	(%r9,%r15), %r12d
	movslq	%r12d, %r12
	vmovdqu	(%rdi,%r12), %xmm11
	leal	(%r9,%r15), %r12d
	addl	$16, %r12d
	movslq	%r12d, %r12
	vinserti128	$1, (%rdi,%r12), %ymm11, %ymm11
	vpunpckhbw	%ymm11, %ymm11, %ymm12  # ymm12 = ymm11[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm12, %ymm8, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vpunpcklbw	%ymm11, %ymm11, %ymm11  # ymm11 = ymm11[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm11, %ymm9, %ymm11
	vpand	%ymm1, %ymm11, %ymm11
	vpackuswb	%ymm12, %ymm11, %ymm11
	leal	32(%r9,%r15), %r12d
	movslq	%r12d, %r12
	vpmovzxbw	(%rdi,%r12), %ymm12     # ymm12 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm12, %ymm10, %ymm12
	vpand	%ymm1, %ymm12, %ymm12
	vextracti128	$1, %ymm12, %xmm13
	vpackuswb	%xmm13, %xmm12, %xmm12
	vextracti128	$1, %ymm11, %xmm14
	vpaddb	%xmm14, %xmm11, %xmm11
	vpshufd	$238, %xmm11, %xmm14            # xmm14 = xmm11[2,3,2,3]
	vpaddb	%xmm14, %xmm11, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %r12d
	vpackuswb	%xmm13, %xmm13, %xmm11
	vpaddb	%xmm11, %xmm12, %xmm11
	vpsadbw	%xmm4, %xmm11, %xmm11
	vmovd	%xmm11, %r13d
	addb	%r12b, %r13b
	addb	%bpl, %r13b
	movb	%r13b, (%r14)
	addq	$16, %r15
	incq	%r14
	cmpq	%r15, %r10
	jne	.LBB0_4
# %bb.5:                                # %for.cond4.for.cond.cleanup6_crit_edge.us
                                        #   in Loop: Header=BB0_3 Depth=1
	incq	%rbx
	addq	%r8, %r9
	addq	%r8, %rdx
	addq	%r8, %r11
	addq	%rax, %rsi
	cmpq	%rcx, %rbx
	jne	.LBB0_3
# %bb.6:
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	.cfi_restore %rbx
	.cfi_restore %r12
	.cfi_restore %r13
	.cfi_restore %r14
	.cfi_restore %r15
	.cfi_restore %rbp
.LBB0_7:                                # %for.cond.cleanup
	vzeroupper
	retq
.Lfunc_end0:
	.size	conv_2d, .Lfunc_end0-conv_2d
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 17.0.1 (https://github.com/llvm/llvm-project.git e19b7dc36bc047b9eb72078d034596be766da350)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
