	.text
	.file	"kernel.c"
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
	subq	$168, %rsp
	.cfi_def_cfa_offset 224
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%ecx, %eax
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rdi, 48(%rsp)                  # 8-byte Spill
	movl	240(%rsp), %edi
	movl	232(%rsp), %ecx
	movl	224(%rsp), %r10d
	movl	%eax, 28(%rsp)                  # 4-byte Spill
	subl	%r10d, %eax
	cltd
	idivl	%edi
	movl	%eax, %r15d
	movl	%r8d, 12(%rsp)                  # 4-byte Spill
	movl	%r8d, %eax
	subl	%ecx, %eax
	cltd
	idivl	%edi
	testl	%r15d, %r15d
	js	.LBB0_62
# %bb.1:                                # %for.cond4.preheader.lr.ph
	testl	%eax, %eax
	js	.LBB0_62
# %bb.2:                                # %for.cond4.preheader.lr.ph.split
	incl	%eax
	testl	%r10d, %r10d
	movq	%rsi, 32(%rsp)                  # 8-byte Spill
	jle	.LBB0_3
# %bb.5:                                # %for.cond4.preheader.lr.ph.split.split.us
	testl	%ecx, %ecx
	jle	.LBB0_6
# %bb.9:                                # %for.cond4.preheader.lr.ph.split.split.us.split.us
	testl	%r9d, %r9d
	jle	.LBB0_10
# %bb.13:                               # %for.cond4.preheader.us170.us.us.preheader
	movslq	12(%rsp), %rdx                  # 4-byte Folded Reload
	movslq	%edi, %rsi
	movslq	%eax, %rdi
	movq	%rdi, 64(%rsp)                  # 8-byte Spill
	incl	%r15d
	movl	%eax, %eax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movl	%r10d, %eax
	movq	%rax, 144(%rsp)                 # 8-byte Spill
	movl	%ecx, %r13d
	movl	%r9d, %ebp
	movl	%ebp, %r14d
	andl	$-128, %r14d
	movl	%ebp, %r8d
	andl	$-16, %r8d
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$96, %rax
	movq	%rax, 128(%rsp)                 # 8-byte Spill
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$96, %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	%rsi, %rax
	movq	%r13, %rcx
	imulq	%rbp, %rcx
	movq	%rcx, 136(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	vpbroadcastw	.LCPI0_1(%rip), %ymm0   # ymm0 = [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]
	vpxor	%xmm1, %xmm1, %xmm1
	movq	%r15, 72(%rsp)                  # 8-byte Spill
	movq	%rsi, 104(%rsp)                 # 8-byte Spill
	jmp	.LBB0_14
	.p2align	4, 0x90
.LBB0_20:                               # %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us
                                        #   in Loop: Header=BB0_14 Depth=1
	movq	80(%rsp), %rcx                  # 8-byte Reload
	incq	%rcx
	movq	72(%rsp), %r15                  # 8-byte Reload
	cmpq	%r15, %rcx
	je	.LBB0_62
.LBB0_14:                               # %for.cond4.preheader.us170.us.us
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_15 Depth 2
                                        #       Child Loop BB0_16 Depth 3
                                        #         Child Loop BB0_22 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #           Child Loop BB0_34 Depth 5
                                        #           Child Loop BB0_36 Depth 5
	movq	%rcx, %rsi
	imulq	%rax, %rsi
	movq	%rsi, 152(%rsp)                 # 8-byte Spill
	movq	%rcx, 80(%rsp)                  # 8-byte Spill
	imulq	64(%rsp), %rcx                  # 8-byte Folded Reload
	movq	%rcx, 112(%rsp)                 # 8-byte Spill
	xorl	%ecx, %ecx
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	xorl	%r10d, %r10d
	jmp	.LBB0_15
	.p2align	4, 0x90
.LBB0_19:                               # %for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us
                                        #   in Loop: Header=BB0_15 Depth=2
	movq	112(%rsp), %rax                 # 8-byte Reload
	movq	120(%rsp), %r10                 # 8-byte Reload
	addq	%r10, %rax
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movb	%r15b, (%rcx,%rax)
	incq	%r10
	movq	104(%rsp), %rax                 # 8-byte Reload
	addq	%rax, 56(%rsp)                  # 8-byte Folded Spill
	cmpq	96(%rsp), %r10                  # 8-byte Folded Reload
	je	.LBB0_20
.LBB0_15:                               # %for.cond8.preheader.us.us.us.us.us.us
                                        #   Parent Loop BB0_14 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_16 Depth 3
                                        #         Child Loop BB0_22 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #           Child Loop BB0_34 Depth 5
                                        #           Child Loop BB0_36 Depth 5
	movq	%r10, 120(%rsp)                 # 8-byte Spill
	imulq	%rax, %r10
	xorl	%ecx, %ecx
	movq	40(%rsp), %rdi                  # 8-byte Reload
	movq	88(%rsp), %r11                  # 8-byte Reload
	xorl	%r15d, %r15d
	jmp	.LBB0_16
	.p2align	4, 0x90
.LBB0_18:                               # %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us
                                        #   in Loop: Header=BB0_16 Depth=3
	movq	(%rsp), %rcx                    # 8-byte Reload
	incq	%rcx
	movq	136(%rsp), %rax                 # 8-byte Reload
	movq	160(%rsp), %r11                 # 8-byte Reload
	addq	%rax, %r11
	movq	16(%rsp), %rdi                  # 8-byte Reload
	addq	%rax, %rdi
	cmpq	144(%rsp), %rcx                 # 8-byte Folded Reload
	je	.LBB0_19
.LBB0_16:                               # %for.cond12.preheader.us.us.us.us.us.us.us.us
                                        #   Parent Loop BB0_14 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB0_22 Depth 4
                                        #           Child Loop BB0_30 Depth 5
                                        #           Child Loop BB0_34 Depth 5
                                        #           Child Loop BB0_36 Depth 5
	movq	152(%rsp), %rax                 # 8-byte Reload
	addq	%rcx, %rax
	testl	%eax, %eax
	movq	%rcx, (%rsp)                    # 8-byte Spill
	movq	%rdi, 16(%rsp)                  # 8-byte Spill
	movq	%r11, 160(%rsp)                 # 8-byte Spill
	js	.LBB0_18
# %bb.17:                               # %for.cond12.preheader.us.us.us.us.us.us.us.us
                                        #   in Loop: Header=BB0_16 Depth=3
	cmpl	28(%rsp), %eax                  # 4-byte Folded Reload
	jge	.LBB0_18
# %bb.21:                               # %for.body15.us.us102.us.us.us.us.us.us.us.us.preheader
                                        #   in Loop: Header=BB0_16 Depth=3
	imull	12(%rsp), %eax                  # 4-byte Folded Reload
	movslq	%eax, %rcx
	addq	56(%rsp), %rcx                  # 8-byte Folded Reload
	imulq	%rbp, %rcx
	movq	128(%rsp), %rax                 # 8-byte Reload
	leaq	(%rax,%rcx), %r12
	addq	48(%rsp), %rcx                  # 8-byte Folded Reload
	xorl	%ebx, %ebx
	jmp	.LBB0_22
	.p2align	4, 0x90
.LBB0_24:                               # %if.end.us.us105.us.us.us.us.us.us.us.us
                                        #   in Loop: Header=BB0_22 Depth=4
	incq	%rbx
	addq	%rbp, %r12
	addq	%rbp, %r11
	addq	%rbp, %rcx
	addq	%rbp, %rdi
	cmpq	%r13, %rbx
	je	.LBB0_18
.LBB0_22:                               # %for.body15.us.us102.us.us.us.us.us.us.us.us
                                        #   Parent Loop BB0_14 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_16 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB0_30 Depth 5
                                        #           Child Loop BB0_34 Depth 5
                                        #           Child Loop BB0_36 Depth 5
	movq	%rbx, %rax
	addq	%r10, %rax
	js	.LBB0_24
# %bb.23:                               # %for.body15.us.us102.us.us.us.us.us.us.us.us
                                        #   in Loop: Header=BB0_22 Depth=4
	cmpq	%rdx, %rax
	jge	.LBB0_24
# %bb.25:                               # %iter.check304
                                        #   in Loop: Header=BB0_22 Depth=4
	cmpl	$16, %r9d
	jae	.LBB0_27
# %bb.26:                               #   in Loop: Header=BB0_22 Depth=4
	xorl	%esi, %esi
	jmp	.LBB0_36
	.p2align	4, 0x90
.LBB0_27:                               # %vector.main.loop.iter.check306
                                        #   in Loop: Header=BB0_22 Depth=4
	cmpl	$128, %r9d
	jae	.LBB0_29
# %bb.28:                               #   in Loop: Header=BB0_22 Depth=4
	xorl	%eax, %eax
	jmp	.LBB0_33
.LBB0_29:                               # %vector.ph307
                                        #   in Loop: Header=BB0_22 Depth=4
	movzbl	%r15b, %eax
	vmovd	%eax, %xmm2
	vpxor	%xmm3, %xmm3, %xmm3
	xorl	%eax, %eax
	vpxor	%xmm4, %xmm4, %xmm4
	vpxor	%xmm5, %xmm5, %xmm5
	.p2align	4, 0x90
.LBB0_30:                               # %vector.body311
                                        #   Parent Loop BB0_14 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_16 Depth=3
                                        #         Parent Loop BB0_22 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vmovdqu	-96(%r12,%rax), %ymm8
	vmovdqu	-64(%r12,%rax), %ymm9
	vmovdqu	-32(%r12,%rax), %ymm10
	vmovdqu	(%r12,%rax), %ymm6
	vmovdqu	-96(%r11,%rax), %ymm11
	vmovdqu	-64(%r11,%rax), %ymm12
	vmovdqu	-32(%r11,%rax), %ymm13
	vmovdqu	(%r11,%rax), %ymm7
	vpunpckhbw	%ymm8, %ymm8, %ymm14    # ymm14 = ymm8[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpunpckhbw	%ymm11, %ymm11, %ymm15  # ymm15 = ymm11[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm14, %ymm15, %ymm14
	vpand	%ymm0, %ymm14, %ymm14
	vpunpcklbw	%ymm8, %ymm8, %ymm8     # ymm8 = ymm8[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpunpcklbw	%ymm11, %ymm11, %ymm11  # ymm11 = ymm11[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm8, %ymm11, %ymm8
	vpand	%ymm0, %ymm8, %ymm8
	vpackuswb	%ymm14, %ymm8, %ymm8
	vpaddb	%ymm2, %ymm8, %ymm2
	vpunpckhbw	%ymm9, %ymm9, %ymm8     # ymm8 = ymm9[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpunpckhbw	%ymm12, %ymm12, %ymm11  # ymm11 = ymm12[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm8, %ymm11, %ymm8
	vpand	%ymm0, %ymm8, %ymm8
	vpunpcklbw	%ymm9, %ymm9, %ymm9     # ymm9 = ymm9[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpunpcklbw	%ymm12, %ymm12, %ymm11  # ymm11 = ymm12[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm9, %ymm11, %ymm9
	vpand	%ymm0, %ymm9, %ymm9
	vpackuswb	%ymm8, %ymm9, %ymm8
	vpaddb	%ymm3, %ymm8, %ymm3
	vpunpckhbw	%ymm10, %ymm10, %ymm8   # ymm8 = ymm10[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpunpckhbw	%ymm13, %ymm13, %ymm9   # ymm9 = ymm13[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm8, %ymm9, %ymm8
	vpand	%ymm0, %ymm8, %ymm8
	vpunpcklbw	%ymm10, %ymm10, %ymm9   # ymm9 = ymm10[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpunpcklbw	%ymm13, %ymm13, %ymm10  # ymm10 = ymm13[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm9, %ymm10, %ymm9
	vpand	%ymm0, %ymm9, %ymm9
	vpackuswb	%ymm8, %ymm9, %ymm8
	vpaddb	%ymm4, %ymm8, %ymm4
	vpunpckhbw	%ymm6, %ymm6, %ymm8     # ymm8 = ymm6[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpunpckhbw	%ymm7, %ymm7, %ymm9     # ymm9 = ymm7[8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,24,24,25,25,26,26,27,27,28,28,29,29,30,30,31,31]
	vpmullw	%ymm8, %ymm9, %ymm8
	vpand	%ymm0, %ymm8, %ymm8
	vpunpcklbw	%ymm6, %ymm6, %ymm6     # ymm6 = ymm6[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpunpcklbw	%ymm7, %ymm7, %ymm7     # ymm7 = ymm7[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,16,16,17,17,18,18,19,19,20,20,21,21,22,22,23,23]
	vpmullw	%ymm6, %ymm7, %ymm6
	vpand	%ymm0, %ymm6, %ymm6
	vpackuswb	%ymm8, %ymm6, %ymm6
	vpaddb	%ymm5, %ymm6, %ymm5
	subq	$-128, %rax
	cmpq	%rax, %r14
	jne	.LBB0_30
# %bb.31:                               # %middle.block301
                                        #   in Loop: Header=BB0_22 Depth=4
	vpaddb	%ymm2, %ymm3, %ymm2
	vpaddb	%ymm2, %ymm4, %ymm2
	vpaddb	%ymm2, %ymm5, %ymm2
	vextracti128	$1, %ymm2, %xmm3
	vpaddb	%xmm3, %xmm2, %xmm2
	vpshufd	$238, %xmm2, %xmm3              # xmm3 = xmm2[2,3,2,3]
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %r15d
	cmpq	%rbp, %r14
	je	.LBB0_24
# %bb.32:                               # %vec.epilog.iter.check328
                                        #   in Loop: Header=BB0_22 Depth=4
	movq	%r14, %rax
	movq	%r14, %rsi
	testb	$112, %bpl
	je	.LBB0_36
.LBB0_33:                               # %vec.epilog.ph329
                                        #   in Loop: Header=BB0_22 Depth=4
	movzbl	%r15b, %esi
	vmovd	%esi, %xmm2
	.p2align	4, 0x90
.LBB0_34:                               # %vec.epilog.vector.body337
                                        #   Parent Loop BB0_14 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_16 Depth=3
                                        #         Parent Loop BB0_22 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	vpmovzxbw	(%rcx,%rax), %ymm3      # ymm3 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmovzxbw	(%rdi,%rax), %ymm4      # ymm4 = mem[0],zero,mem[1],zero,mem[2],zero,mem[3],zero,mem[4],zero,mem[5],zero,mem[6],zero,mem[7],zero,mem[8],zero,mem[9],zero,mem[10],zero,mem[11],zero,mem[12],zero,mem[13],zero,mem[14],zero,mem[15],zero
	vpmullw	%ymm3, %ymm4, %ymm3
	vpand	%ymm0, %ymm3, %ymm3
	vextracti128	$1, %ymm3, %xmm4
	vpackuswb	%xmm4, %xmm3, %xmm3
	vpaddb	%xmm2, %xmm3, %xmm2
	addq	$16, %rax
	cmpq	%rax, %r8
	jne	.LBB0_34
# %bb.35:                               # %vec.epilog.middle.block326
                                        #   in Loop: Header=BB0_22 Depth=4
	vpshufd	$238, %xmm2, %xmm3              # xmm3 = xmm2[2,3,2,3]
	vpaddb	%xmm3, %xmm2, %xmm2
	vpsadbw	%xmm1, %xmm2, %xmm2
	vmovd	%xmm2, %r15d
	movq	%r8, %rsi
	cmpq	%rbp, %r8
	je	.LBB0_24
	.p2align	4, 0x90
.LBB0_36:                               # %for.body28.us.us.us.us.us.us.us.us.us.us
                                        #   Parent Loop BB0_14 Depth=1
                                        #     Parent Loop BB0_15 Depth=2
                                        #       Parent Loop BB0_16 Depth=3
                                        #         Parent Loop BB0_22 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	movzbl	(%rdi,%rsi), %eax
	mulb	(%rcx,%rsi)
	addb	%al, %r15b
	incq	%rsi
	cmpq	%rsi, %rbp
	jne	.LBB0_36
	jmp	.LBB0_24
.LBB0_3:                                # %for.cond4.preheader.preheader
	movslq	%eax, %rbx
	movl	%ebx, %r14d
	leal	1(%r15), %r13d
	movq	%r13, (%rsp)                    # 8-byte Spill
                                        # kill: def $r13d killed $r13d killed $r13 def $r13
	andl	$7, %r13d
	cmpl	$7, %r15d
	jae	.LBB0_57
# %bb.4:
	xorl	%ebp, %ebp
	jmp	.LBB0_59
.LBB0_6:                                # %for.cond4.preheader.us170.preheader
	movslq	%eax, %rcx
	incl	%r15d
	movl	%eax, %edx
	movl	%edx, %ebx
	andl	$-128, %ebx
	movl	%edx, %edi
	andl	$-16, %edi
	leaq	96(%rsi), %r8
	xorl	%r9d, %r9d
	vpxor	%xmm0, %xmm0, %xmm0
	vpxor	%xmm1, %xmm1, %xmm1
	jmp	.LBB0_7
	.p2align	4, 0x90
.LBB0_56:                               # %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181
                                        #   in Loop: Header=BB0_7 Depth=1
	incq	%r9
	addq	%rcx, %r8
	addq	%rcx, %rsi
	cmpq	%r15, %r9
	je	.LBB0_62
.LBB0_7:                                # %iter.check
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_50 Depth 2
                                        #     Child Loop BB0_53 Depth 2
                                        #     Child Loop BB0_55 Depth 2
	cmpl	$16, %eax
	jae	.LBB0_47
# %bb.8:                                #   in Loop: Header=BB0_7 Depth=1
	xorl	%r10d, %r10d
	jmp	.LBB0_55
	.p2align	4, 0x90
.LBB0_47:                               # %vector.main.loop.iter.check
                                        #   in Loop: Header=BB0_7 Depth=1
	cmpl	$128, %eax
	jae	.LBB0_49
# %bb.48:                               #   in Loop: Header=BB0_7 Depth=1
	xorl	%r11d, %r11d
	jmp	.LBB0_53
	.p2align	4, 0x90
.LBB0_49:                               # %vector.body.preheader
                                        #   in Loop: Header=BB0_7 Depth=1
	xorl	%r10d, %r10d
	.p2align	4, 0x90
.LBB0_50:                               # %vector.body
                                        #   Parent Loop BB0_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	%ymm1, -96(%r8,%r10)
	vmovdqu	%ymm1, -64(%r8,%r10)
	vmovdqu	%ymm1, -32(%r8,%r10)
	vmovdqu	%ymm1, (%r8,%r10)
	subq	$-128, %r10
	cmpq	%r10, %rbx
	jne	.LBB0_50
# %bb.51:                               # %middle.block
                                        #   in Loop: Header=BB0_7 Depth=1
	cmpq	%rdx, %rbx
	je	.LBB0_56
# %bb.52:                               # %vec.epilog.iter.check
                                        #   in Loop: Header=BB0_7 Depth=1
	movq	%rbx, %r11
	movq	%rbx, %r10
	testb	$112, %dl
	je	.LBB0_55
	.p2align	4, 0x90
.LBB0_53:                               # %vec.epilog.vector.body
                                        #   Parent Loop BB0_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	%xmm0, (%rsi,%r11)
	addq	$16, %r11
	cmpq	%r11, %rdi
	jne	.LBB0_53
# %bb.54:                               # %vec.epilog.middle.block
                                        #   in Loop: Header=BB0_7 Depth=1
	movq	%rdi, %r10
	cmpq	%rdx, %rdi
	je	.LBB0_56
	.p2align	4, 0x90
.LBB0_55:                               # %for.cond8.preheader.us.us174
                                        #   Parent Loop BB0_7 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movb	$0, (%rsi,%r10)
	incq	%r10
	cmpq	%r10, %rdx
	jne	.LBB0_55
	jmp	.LBB0_56
.LBB0_57:                               # %for.cond4.preheader.preheader.new
	movq	(%rsp), %rax                    # 8-byte Reload
	andl	$-8, %eax
	movq	%rax, (%rsp)                    # 8-byte Spill
	leaq	(,%rbx,8), %rax
	movq	%rax, 16(%rsp)                  # 8-byte Spill
	xorl	%ebp, %ebp
	movq	%rsi, %r15
	.p2align	4, 0x90
.LBB0_58:                               # %for.cond4.preheader
                                        # =>This Inner Loop Header: Depth=1
	movq	%r15, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	leaq	(%r15,%rbx), %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r12
	movq	%r12, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	$8, %rbp
	addq	16(%rsp), %r15                  # 8-byte Folded Reload
	cmpq	%rbp, (%rsp)                    # 8-byte Folded Reload
	jne	.LBB0_58
.LBB0_59:                               # %for.cond.cleanup.loopexit350.unr-lcssa
	testq	%r13, %r13
	movq	32(%rsp), %r15                  # 8-byte Reload
	je	.LBB0_62
# %bb.60:                               # %for.cond4.preheader.epil.preheader
	imulq	%rbx, %rbp
	addq	%rbp, %r15
	.p2align	4, 0x90
.LBB0_61:                               # %for.cond4.preheader.epil
                                        # =>This Inner Loop Header: Depth=1
	movq	%r15, %rdi
	xorl	%esi, %esi
	movq	%r14, %rdx
	callq	memset@PLT
	addq	%rbx, %r15
	decq	%r13
	jne	.LBB0_61
.LBB0_62:                               # %for.cond.cleanup
	addq	$168, %rsp
	.cfi_def_cfa_offset 56
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
.LBB0_10:                               # %for.cond4.preheader.us170.us.preheader
	.cfi_def_cfa_offset 224
	movslq	%eax, %rcx
	incl	%r15d
	movl	%eax, %edx
	movl	%edx, %ebx
	andl	$-128, %ebx
	movl	%edx, %edi
	andl	$-16, %edi
	leaq	96(%rsi), %r8
	xorl	%r9d, %r9d
	vpxor	%xmm0, %xmm0, %xmm0
	vpxor	%xmm1, %xmm1, %xmm1
	jmp	.LBB0_11
	.p2align	4, 0x90
.LBB0_46:                               # %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us
                                        #   in Loop: Header=BB0_11 Depth=1
	incq	%r9
	addq	%rcx, %r8
	addq	%rcx, %rsi
	cmpq	%r15, %r9
	je	.LBB0_62
.LBB0_11:                               # %iter.check277
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_40 Depth 2
                                        #     Child Loop BB0_43 Depth 2
                                        #     Child Loop BB0_45 Depth 2
	cmpl	$16, %eax
	jae	.LBB0_37
# %bb.12:                               #   in Loop: Header=BB0_11 Depth=1
	xorl	%r10d, %r10d
	jmp	.LBB0_45
	.p2align	4, 0x90
.LBB0_37:                               # %vector.main.loop.iter.check279
                                        #   in Loop: Header=BB0_11 Depth=1
	cmpl	$128, %eax
	jae	.LBB0_39
# %bb.38:                               #   in Loop: Header=BB0_11 Depth=1
	xorl	%r11d, %r11d
	jmp	.LBB0_43
	.p2align	4, 0x90
.LBB0_39:                               # %vector.body284.preheader
                                        #   in Loop: Header=BB0_11 Depth=1
	xorl	%r10d, %r10d
	.p2align	4, 0x90
.LBB0_40:                               # %vector.body284
                                        #   Parent Loop BB0_11 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	%ymm1, -96(%r8,%r10)
	vmovdqu	%ymm1, -64(%r8,%r10)
	vmovdqu	%ymm1, -32(%r8,%r10)
	vmovdqu	%ymm1, (%r8,%r10)
	subq	$-128, %r10
	cmpq	%r10, %rbx
	jne	.LBB0_40
# %bb.41:                               # %middle.block274
                                        #   in Loop: Header=BB0_11 Depth=1
	cmpq	%rdx, %rbx
	je	.LBB0_46
# %bb.42:                               # %vec.epilog.iter.check289
                                        #   in Loop: Header=BB0_11 Depth=1
	movq	%rbx, %r11
	movq	%rbx, %r10
	testb	$112, %dl
	je	.LBB0_45
	.p2align	4, 0x90
.LBB0_43:                               # %vec.epilog.vector.body298
                                        #   Parent Loop BB0_11 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovdqu	%xmm0, (%rsi,%r11)
	addq	$16, %r11
	cmpq	%r11, %rdi
	jne	.LBB0_43
# %bb.44:                               # %vec.epilog.middle.block287
                                        #   in Loop: Header=BB0_11 Depth=1
	movq	%rdi, %r10
	cmpq	%rdx, %rdi
	je	.LBB0_46
	.p2align	4, 0x90
.LBB0_45:                               # %for.cond8.preheader.us.us.us183.us
                                        #   Parent Loop BB0_11 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movb	$0, (%rsi,%r10)
	incq	%r10
	cmpq	%r10, %rdx
	jne	.LBB0_45
	jmp	.LBB0_46
.Lfunc_end0:
	.size	conv_2d, .Lfunc_end0-conv_2d
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 17.0.1 (https://github.com/llvm/llvm-project.git e19b7dc36bc047b9eb72078d034596be766da350)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
