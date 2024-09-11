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
                                        # kill: def $r8d killed $r8d def $r8
                                        # kill: def $ecx killed $ecx def $rcx
	movq	%rcx, -8(%rsp)                  # 8-byte Spill
	cmpl	$5, %ecx
	jl	.LBB0_8
# %bb.1:                                # %entry
	cmpl	$5, %r8d
	jl	.LBB0_8
# %bb.2:                                # %for.cond4.preheader.us.preheader
	leal	-4(%r8), %eax
	movq	-8(%rsp), %rcx                  # 8-byte Reload
	addl	$-4, %ecx
	movq	%rcx, -8(%rsp)                  # 8-byte Spill
	shll	$4, %r8d
	movl	$64, %ecx
	xorl	%r10d, %r10d
	vpbroadcastw	.LCPI0_1(%rip), %ymm0   # ymm0 = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
	vpxor	%xmm1, %xmm1, %xmm1
	.p2align	4, 0x90
.LBB0_3:                                # %for.cond4.preheader.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_6 Depth 2
                                        #       Child Loop BB0_4 Depth 3
	movq	%r10, %r11
	imulq	%rax, %r11
	movl	%ecx, %ebx
	xorl	%r14d, %r14d
	.p2align	4, 0x90
.LBB0_6:                                # %for.cond8.preheader.us
                                        #   Parent Loop BB0_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_4 Depth 3
	movl	$64, %r15d
	xorl	%ebp, %ebp
	movl	%ebx, %r12d
	.p2align	4, 0x90
.LBB0_4:                                # %for.cond12.preheader.us
                                        #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	leal	-64(%r12), %r9d
	movslq	%r9d, %r9
	vpmovzxbw	(%rdi,%r9), %ymm2       # ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	-64(%rdx,%r15), %ymm3   # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm2, %ymm3, %ymm2
	vpand	%ymm0, %ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackuswb	%xmm3, %xmm2, %xmm2
	vpackuswb	%xmm3, %xmm3, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %r13d
	addb	%bpl, %r13b
	leal	-48(%r12), %r9d
	movslq	%r9d, %r9
	vpmovzxbw	(%rdi,%r9), %ymm2       # ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	-48(%rdx,%r15), %ymm3   # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm2, %ymm3, %ymm2
	vpand	%ymm0, %ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackuswb	%xmm3, %xmm2, %xmm2
	vpackuswb	%xmm3, %xmm3, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %ebp
	leal	-32(%r12), %r9d
	movslq	%r9d, %r9
	vpmovzxbw	(%rdi,%r9), %ymm2       # ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	-32(%rdx,%r15), %ymm3   # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm2, %ymm3, %ymm2
	vpand	%ymm0, %ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackuswb	%xmm3, %xmm2, %xmm2
	vpackuswb	%xmm3, %xmm3, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %r9d
	addb	%bpl, %r9b
	addb	%r13b, %r9b
	leal	-16(%r12), %ebp
	movslq	%ebp, %r13
	vpmovzxbw	(%rdi,%r13), %ymm2      # ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	-16(%rdx,%r15), %ymm3   # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm2, %ymm3, %ymm2
	vpand	%ymm0, %ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackuswb	%xmm3, %xmm2, %xmm2
	vpackuswb	%xmm3, %xmm3, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %r13d
	movslq	%r12d, %r12
	vpmovzxbw	(%rdi,%r12), %ymm2      # ymm2 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	(%rdx,%r15), %ymm3      # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm2, %ymm3, %ymm2
	vpand	%ymm0, %ymm2, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpackuswb	%xmm3, %xmm2, %xmm2
	vpackuswb	%xmm3, %xmm3, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %ebp
	addb	%r13b, %bpl
	addb	%r9b, %bpl
	addq	$80, %r15
	addl	%r8d, %r12d
	cmpq	$464, %r15                      # imm = 0x1D0
	jne	.LBB0_4
# %bb.5:                                # %for.cond.cleanup10.us
                                        #   in Loop: Header=BB0_6 Depth=2
	leaq	(%r14,%r11), %r9
	movb	%bpl, (%rsi,%r9)
	incq	%r14
	addl	$16, %ebx
	cmpq	%rax, %r14
	jne	.LBB0_6
# %bb.7:                                # %for.cond4.for.cond.cleanup6_crit_edge.us
                                        #   in Loop: Header=BB0_3 Depth=1
	incq	%r10
	addl	%r8d, %ecx
	cmpq	-8(%rsp), %r10                  # 8-byte Folded Reload
	jne	.LBB0_3
.LBB0_8:                                # %for.cond.cleanup
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
	vzeroupper
	retq
.Lfunc_end0:
	.size	conv_2d, .Lfunc_end0-conv_2d
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 17.0.1 (https://github.com/llvm/llvm-project.git e19b7dc36bc047b9eb72078d034596be766da350)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
