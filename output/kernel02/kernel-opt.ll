; ModuleID = '/home/csalvado/tensor-layout/yaconv-update/kernels/kernel02/kernel-opt.c'
source_filename = "/home/csalvado/tensor-layout/yaconv-update/kernels/kernel02/kernel-opt.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @conv_2d(ptr noalias nocapture noundef readonly %input, ptr noalias nocapture noundef writeonly %output, ptr noalias nocapture noundef readonly %filter, i32 noundef %input_height, i32 noundef %input_width, i32 noundef %depth, i32 noundef %filter_height, i32 noundef %filter_width, i32 noundef %stride) local_unnamed_addr #0 {
entry:
  %cmp100 = icmp sgt i32 %input_height, 4
  %cmp598 = icmp sgt i32 %input_width, 4
  %or.cond = and i1 %cmp100, %cmp598
  br i1 %or.cond, label %for.cond4.preheader.us.preheader, label %for.cond.cleanup

for.cond4.preheader.us.preheader:                 ; preds = %entry
  %add3 = add i32 %input_width, -4
  %add = add i32 %input_height, -4
  %0 = zext i32 %add3 to i64
  %wide.trip.count122 = zext i32 %add to i64
  %wide.trip.count = zext i32 %add3 to i64
  br label %for.cond4.preheader.us

for.cond4.preheader.us:                           ; preds = %for.cond4.preheader.us.preheader, %for.cond4.for.cond.cleanup6_crit_edge.us
  %indvars.iv118 = phi i64 [ 0, %for.cond4.preheader.us.preheader ], [ %indvars.iv.next119, %for.cond4.for.cond.cleanup6_crit_edge.us ]
  %1 = mul nsw i64 %indvars.iv118, %0
  br label %for.cond8.preheader.us

for.cond.cleanup10.us:                            ; preds = %for.cond12.preheader.us
  %2 = add nuw nsw i64 %indvars.iv113, %1
  %arrayidx53.us = getelementptr inbounds i8, ptr %output, i64 %2
  store i8 %op.rdx.4, ptr %arrayidx53.us, align 1, !tbaa !5
  %indvars.iv.next114 = add nuw nsw i64 %indvars.iv113, 1
  %exitcond117.not = icmp eq i64 %indvars.iv.next114, %wide.trip.count
  br i1 %exitcond117.not, label %for.cond4.for.cond.cleanup6_crit_edge.us, label %for.cond8.preheader.us, !llvm.loop !8

for.cond12.preheader.us:                          ; preds = %for.cond8.preheader.us, %for.cond12.preheader.us
  %indvars.iv107 = phi i64 [ 0, %for.cond8.preheader.us ], [ %indvars.iv.next108, %for.cond12.preheader.us ]
  %sum_block.096.us = phi i8 [ 0, %for.cond8.preheader.us ], [ %op.rdx.4, %for.cond12.preheader.us ]
  %3 = add nuw nsw i64 %indvars.iv107, %indvars.iv118
  %4 = trunc i64 %3 to i32
  %5 = mul i32 %4, %input_width
  %add18.us = add i32 %5, %40
  %mul31.us = shl nsw i32 %add18.us, 4
  %6 = mul i64 %indvars.iv107, 80
  %7 = sext i32 %mul31.us to i64
  %arrayidx.us = getelementptr inbounds i8, ptr %input, i64 %7
  %arrayidx38.us = getelementptr inbounds i8, ptr %filter, i64 %6
  %8 = load <16 x i8>, ptr %arrayidx.us, align 1, !tbaa !5
  %9 = load <16 x i8>, ptr %arrayidx38.us, align 1, !tbaa !5
  %10 = mul <16 x i8> %9, %8
  %11 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %10)
  %op.rdx = add i8 %11, %sum_block.096.us
  %add30.us.1 = shl i32 %add18.us, 4
  %mul31.us.1 = add i32 %add30.us.1, 16
  %12 = mul i64 %indvars.iv107, 80
  %13 = add i64 %12, 16
  %14 = sext i32 %mul31.us.1 to i64
  %arrayidx.us.1 = getelementptr inbounds i8, ptr %input, i64 %14
  %arrayidx38.us.1 = getelementptr inbounds i8, ptr %filter, i64 %13
  %15 = load <16 x i8>, ptr %arrayidx.us.1, align 1, !tbaa !5
  %16 = load <16 x i8>, ptr %arrayidx38.us.1, align 1, !tbaa !5
  %17 = mul <16 x i8> %16, %15
  %18 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %17)
  %op.rdx.1 = add i8 %18, %op.rdx
  %add30.us.2 = shl i32 %add18.us, 4
  %mul31.us.2 = add i32 %add30.us.2, 32
  %19 = mul i64 %indvars.iv107, 80
  %20 = add i64 %19, 32
  %21 = sext i32 %mul31.us.2 to i64
  %arrayidx.us.2 = getelementptr inbounds i8, ptr %input, i64 %21
  %arrayidx38.us.2 = getelementptr inbounds i8, ptr %filter, i64 %20
  %22 = load <16 x i8>, ptr %arrayidx.us.2, align 1, !tbaa !5
  %23 = load <16 x i8>, ptr %arrayidx38.us.2, align 1, !tbaa !5
  %24 = mul <16 x i8> %23, %22
  %25 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %24)
  %op.rdx.2 = add i8 %25, %op.rdx.1
  %add30.us.3 = shl i32 %add18.us, 4
  %mul31.us.3 = add i32 %add30.us.3, 48
  %26 = mul i64 %indvars.iv107, 80
  %27 = add i64 %26, 48
  %28 = sext i32 %mul31.us.3 to i64
  %arrayidx.us.3 = getelementptr inbounds i8, ptr %input, i64 %28
  %arrayidx38.us.3 = getelementptr inbounds i8, ptr %filter, i64 %27
  %29 = load <16 x i8>, ptr %arrayidx.us.3, align 1, !tbaa !5
  %30 = load <16 x i8>, ptr %arrayidx38.us.3, align 1, !tbaa !5
  %31 = mul <16 x i8> %30, %29
  %32 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %31)
  %op.rdx.3 = add i8 %32, %op.rdx.2
  %add30.us.4 = shl i32 %add18.us, 4
  %mul31.us.4 = add i32 %add30.us.4, 64
  %33 = mul i64 %indvars.iv107, 80
  %34 = add i64 %33, 64
  %35 = sext i32 %mul31.us.4 to i64
  %arrayidx.us.4 = getelementptr inbounds i8, ptr %input, i64 %35
  %arrayidx38.us.4 = getelementptr inbounds i8, ptr %filter, i64 %34
  %36 = load <16 x i8>, ptr %arrayidx.us.4, align 1, !tbaa !5
  %37 = load <16 x i8>, ptr %arrayidx38.us.4, align 1, !tbaa !5
  %38 = mul <16 x i8> %37, %36
  %39 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %38)
  %op.rdx.4 = add i8 %39, %op.rdx.3
  %indvars.iv.next108 = add nuw nsw i64 %indvars.iv107, 1
  %exitcond112.not = icmp eq i64 %indvars.iv.next108, 5
  br i1 %exitcond112.not, label %for.cond.cleanup10.us, label %for.cond12.preheader.us, !llvm.loop !10

for.cond8.preheader.us:                           ; preds = %for.cond4.preheader.us, %for.cond.cleanup10.us
  %indvars.iv113 = phi i64 [ 0, %for.cond4.preheader.us ], [ %indvars.iv.next114, %for.cond.cleanup10.us ]
  %40 = trunc i64 %indvars.iv113 to i32
  br label %for.cond12.preheader.us

for.cond4.for.cond.cleanup6_crit_edge.us:         ; preds = %for.cond.cleanup10.us
  %indvars.iv.next119 = add nuw nsw i64 %indvars.iv118, 1
  %exitcond123.not = icmp eq i64 %indvars.iv.next119, %wide.trip.count122
  br i1 %exitcond123.not, label %for.cond.cleanup, label %for.cond4.preheader.us, !llvm.loop !11

for.cond.cleanup:                                 ; preds = %for.cond4.for.cond.cleanup6_crit_edge.us, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>) #1

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="alderlake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avxvnni,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+hreset,+invpcid,+kl,+lzcnt,+mmx,+movbe,+movdir64b,+movdiri,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+ptwrite,+rdpid,+rdrnd,+rdseed,+sahf,+serialize,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+waitpkg,+widekl,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-complex,-amx-fp16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512fp16,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnniint16,-avxvnniint8,-cldemote,-clzero,-cmpccxadd,-enqcmd,-fma4,-lwp,-mwaitx,-prefetchi,-prefetchwt1,-raoint,-rdpru,-rtm,-sgx,-sha512,-sm3,-sm4,-sse4a,-tbm,-tsxldtrk,-uintr,-wbnoinvd,-xop" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 17.0.1 (https://github.com/llvm/llvm-project.git e19b7dc36bc047b9eb72078d034596be766da350)"}
!5 = !{!6, !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = distinct !{!8, !9}
!9 = !{!"llvm.loop.mustprogress"}
!10 = distinct !{!10, !9}
!11 = distinct !{!11, !9}
