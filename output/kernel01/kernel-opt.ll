; ModuleID = '/home/csalvado/tensor-layout/yaconv-update/kernels/kernel01/kernel-opt.c'
source_filename = "/home/csalvado/tensor-layout/yaconv-update/kernels/kernel01/kernel-opt.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @conv_2d(ptr noalias nocapture noundef readonly %input, ptr noalias nocapture noundef writeonly %output, ptr noalias nocapture noundef readonly %filter, i32 noundef %input_height, i32 noundef %input_width, i32 noundef %depth, i32 noundef %filter_height, i32 noundef %filter_width, i32 noundef %stride) local_unnamed_addr #0 {
entry:
  %cmp100 = icmp sgt i32 %input_height, 2
  %cmp598 = icmp sgt i32 %input_width, 2
  %or.cond = and i1 %cmp100, %cmp598
  br i1 %or.cond, label %for.cond4.preheader.us.preheader, label %for.cond.cleanup

for.cond4.preheader.us.preheader:                 ; preds = %entry
  %add3 = add i32 %input_width, -2
  %add = add i32 %input_height, -2
  %0 = zext i32 %add3 to i64
  %wide.trip.count127 = zext i32 %add to i64
  %wide.trip.count = zext i32 %add3 to i64
  %1 = load <32 x i8>, ptr %filter, align 1, !tbaa !5
  %arrayidx38.us.2112 = getelementptr inbounds i8, ptr %filter, i64 32
  %2 = load <16 x i8>, ptr %arrayidx38.us.2112, align 1, !tbaa !5
  %arrayidx38.us.1 = getelementptr inbounds i8, ptr %filter, i64 48
  %3 = load <32 x i8>, ptr %arrayidx38.us.1, align 1, !tbaa !5
  %arrayidx38.us.2112.1 = getelementptr inbounds i8, ptr %filter, i64 80
  %4 = load <16 x i8>, ptr %arrayidx38.us.2112.1, align 1, !tbaa !5
  %arrayidx38.us.2 = getelementptr inbounds i8, ptr %filter, i64 96
  %5 = load <32 x i8>, ptr %arrayidx38.us.2, align 1, !tbaa !5
  %arrayidx38.us.2112.2 = getelementptr inbounds i8, ptr %filter, i64 128
  %6 = load <16 x i8>, ptr %arrayidx38.us.2112.2, align 1, !tbaa !5
  br label %for.cond4.preheader.us

for.cond4.preheader.us:                           ; preds = %for.cond4.preheader.us.preheader, %for.cond4.for.cond.cleanup6_crit_edge.us
  %indvars.iv123 = phi i64 [ 0, %for.cond4.preheader.us.preheader ], [ %indvars.iv.next124, %for.cond4.for.cond.cleanup6_crit_edge.us ]
  %7 = mul nsw i64 %indvars.iv123, %0
  %8 = trunc i64 %indvars.iv123 to i32
  %9 = mul i32 %8, %input_width
  %10 = trunc i64 %indvars.iv123 to i32
  %11 = add i32 %10, 1
  %12 = mul i32 %11, %input_width
  %13 = trunc i64 %indvars.iv123 to i32
  %14 = add i32 %13, 2
  %15 = mul i32 %14, %input_width
  br label %for.cond8.preheader.us

for.cond8.preheader.us:                           ; preds = %for.cond4.preheader.us, %for.cond8.preheader.us
  %indvars.iv118 = phi i64 [ 0, %for.cond4.preheader.us ], [ %indvars.iv.next119, %for.cond8.preheader.us ]
  %16 = trunc i64 %indvars.iv118 to i32
  %add18.us = add i32 %9, %16
  %mul31.us = shl nsw i32 %add18.us, 4
  %17 = sext i32 %mul31.us to i64
  %arrayidx.us = getelementptr inbounds i8, ptr %input, i64 %17
  %18 = load <16 x i8>, ptr %arrayidx.us, align 1, !tbaa !5
  %add30.us.1 = shl i32 %add18.us, 4
  %mul31.us.1 = add i32 %add30.us.1, 16
  %19 = sext i32 %mul31.us.1 to i64
  %arrayidx.us.1107 = getelementptr inbounds i8, ptr %input, i64 %19
  %20 = load <16 x i8>, ptr %arrayidx.us.1107, align 1, !tbaa !5
  %21 = shufflevector <16 x i8> %18, <16 x i8> %20, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %22 = mul <32 x i8> %1, %21
  %add30.us.2 = shl i32 %add18.us, 4
  %mul31.us.2 = add i32 %add30.us.2, 32
  %23 = sext i32 %mul31.us.2 to i64
  %arrayidx.us.2111 = getelementptr inbounds i8, ptr %input, i64 %23
  %24 = load <16 x i8>, ptr %arrayidx.us.2111, align 1, !tbaa !5
  %25 = mul <16 x i8> %2, %24
  %26 = tail call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %22)
  %27 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %25)
  %op.rdx = add i8 %26, %27
  %add18.us.1 = add i32 %12, %16
  %mul31.us.1130 = shl nsw i32 %add18.us.1, 4
  %28 = sext i32 %mul31.us.1130 to i64
  %arrayidx.us.1 = getelementptr inbounds i8, ptr %input, i64 %28
  %29 = load <16 x i8>, ptr %arrayidx.us.1, align 1, !tbaa !5
  %add30.us.1.1 = shl i32 %add18.us.1, 4
  %mul31.us.1.1 = add i32 %add30.us.1.1, 16
  %30 = sext i32 %mul31.us.1.1 to i64
  %arrayidx.us.1107.1 = getelementptr inbounds i8, ptr %input, i64 %30
  %31 = load <16 x i8>, ptr %arrayidx.us.1107.1, align 1, !tbaa !5
  %32 = shufflevector <16 x i8> %29, <16 x i8> %31, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %33 = mul <32 x i8> %3, %32
  %add30.us.2.1 = shl i32 %add18.us.1, 4
  %mul31.us.2.1 = add i32 %add30.us.2.1, 32
  %34 = sext i32 %mul31.us.2.1 to i64
  %arrayidx.us.2111.1 = getelementptr inbounds i8, ptr %input, i64 %34
  %35 = load <16 x i8>, ptr %arrayidx.us.2111.1, align 1, !tbaa !5
  %36 = mul <16 x i8> %4, %35
  %37 = tail call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %33)
  %38 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %36)
  %op.rdx.1 = add i8 %37, %38
  %op.rdx129.1 = add i8 %op.rdx.1, %op.rdx
  %add18.us.2 = add i32 %15, %16
  %mul31.us.2131 = shl nsw i32 %add18.us.2, 4
  %39 = sext i32 %mul31.us.2131 to i64
  %arrayidx.us.2 = getelementptr inbounds i8, ptr %input, i64 %39
  %40 = load <16 x i8>, ptr %arrayidx.us.2, align 1, !tbaa !5
  %add30.us.1.2 = shl i32 %add18.us.2, 4
  %mul31.us.1.2 = add i32 %add30.us.1.2, 16
  %41 = sext i32 %mul31.us.1.2 to i64
  %arrayidx.us.1107.2 = getelementptr inbounds i8, ptr %input, i64 %41
  %42 = load <16 x i8>, ptr %arrayidx.us.1107.2, align 1, !tbaa !5
  %43 = shufflevector <16 x i8> %40, <16 x i8> %42, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %44 = mul <32 x i8> %5, %43
  %add30.us.2.2 = shl i32 %add18.us.2, 4
  %mul31.us.2.2 = add i32 %add30.us.2.2, 32
  %45 = sext i32 %mul31.us.2.2 to i64
  %arrayidx.us.2111.2 = getelementptr inbounds i8, ptr %input, i64 %45
  %46 = load <16 x i8>, ptr %arrayidx.us.2111.2, align 1, !tbaa !5
  %47 = mul <16 x i8> %6, %46
  %48 = tail call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %44)
  %49 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %47)
  %op.rdx.2 = add i8 %48, %49
  %op.rdx129.2 = add i8 %op.rdx.2, %op.rdx129.1
  %50 = add nuw nsw i64 %indvars.iv118, %7
  %arrayidx53.us = getelementptr inbounds i8, ptr %output, i64 %50
  store i8 %op.rdx129.2, ptr %arrayidx53.us, align 1, !tbaa !5
  %indvars.iv.next119 = add nuw nsw i64 %indvars.iv118, 1
  %exitcond122.not = icmp eq i64 %indvars.iv.next119, %wide.trip.count
  br i1 %exitcond122.not, label %for.cond4.for.cond.cleanup6_crit_edge.us, label %for.cond8.preheader.us, !llvm.loop !8

for.cond4.for.cond.cleanup6_crit_edge.us:         ; preds = %for.cond8.preheader.us
  %indvars.iv.next124 = add nuw nsw i64 %indvars.iv123, 1
  %exitcond128.not = icmp eq i64 %indvars.iv.next124, %wide.trip.count127
  br i1 %exitcond128.not, label %for.cond.cleanup, label %for.cond4.preheader.us, !llvm.loop !10

for.cond.cleanup:                                 ; preds = %for.cond4.for.cond.cleanup6_crit_edge.us, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.vector.reduce.add.v32i8(<32 x i8>) #1

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
