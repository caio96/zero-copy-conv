; ModuleID = '/home/csalvado/tensor-layout/yaconv-update/kernels/kernel01/kernel.c'
source_filename = "/home/csalvado/tensor-layout/yaconv-update/kernels/kernel01/kernel.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nofree nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @conv_2d(ptr noalias nocapture noundef readonly %input, ptr noalias nocapture noundef writeonly %output, ptr noalias nocapture noundef readonly %filter, i32 noundef %input_height, i32 noundef %input_width, i32 noundef %depth, i32 noundef %filter_height, i32 noundef %filter_width, i32 noundef %stride) local_unnamed_addr #0 {
entry:
  %sub = sub nsw i32 %input_height, %filter_height
  %div = sdiv i32 %sub, %stride
  %sub1 = sub nsw i32 %input_width, %filter_width
  %div2 = sdiv i32 %sub1, %stride
  %add3 = add i32 %div2, 1
  %cmp.not168 = icmp slt i32 %div, 0
  br i1 %cmp.not168, label %for.cond.cleanup, label %for.cond4.preheader.lr.ph

for.cond4.preheader.lr.ph:                        ; preds = %entry
  %cmp5.not134 = icmp slt i32 %div2, 0
  %cmp1396 = icmp sgt i32 %filter_width, 0
  %cmp2693 = icmp sgt i32 %depth, 0
  br i1 %cmp5.not134, label %for.cond.cleanup, label %for.cond4.preheader.lr.ph.split

for.cond4.preheader.lr.ph.split:                  ; preds = %for.cond4.preheader.lr.ph
  %cmp9111 = icmp sgt i32 %filter_height, 0
  br i1 %cmp9111, label %for.cond4.preheader.lr.ph.split.split.us, label %for.cond4.preheader.preheader

for.cond4.preheader.preheader:                    ; preds = %for.cond4.preheader.lr.ph.split
  %0 = sext i32 %add3 to i64
  %1 = zext i32 %add3 to i64
  %2 = add nuw i32 %div, 1
  %wide.trip.count = zext i32 %2 to i64
  %xtraiter = and i64 %wide.trip.count, 7
  %3 = icmp ult i32 %div, 7
  br i1 %3, label %for.cond.cleanup.loopexit350.unr-lcssa, label %for.cond4.preheader.preheader.new

for.cond4.preheader.preheader.new:                ; preds = %for.cond4.preheader.preheader
  %unroll_iter = and i64 %wide.trip.count, 4294967288
  br label %for.cond4.preheader

for.cond4.preheader.lr.ph.split.split.us:         ; preds = %for.cond4.preheader.lr.ph.split
  br i1 %cmp1396, label %for.cond4.preheader.lr.ph.split.split.us.split.us, label %for.cond4.preheader.us170.preheader

for.cond4.preheader.us170.preheader:              ; preds = %for.cond4.preheader.lr.ph.split.split.us
  %4 = sext i32 %add3 to i64
  %5 = add nuw i32 %div, 1
  %wide.trip.count212 = zext i32 %5 to i64
  %wide.trip.count206 = zext i32 %add3 to i64
  %min.iters.check = icmp ult i32 %add3, 16
  %min.iters.check268 = icmp ult i32 %add3, 128
  %n.vec = and i64 %wide.trip.count206, 4294967168
  %cmp.n = icmp eq i64 %n.vec, %wide.trip.count206
  %n.vec.remaining = and i64 %wide.trip.count206, 112
  %min.epilog.iters.check = icmp eq i64 %n.vec.remaining, 0
  %n.vec270 = and i64 %wide.trip.count206, 4294967280
  %cmp.n271 = icmp eq i64 %n.vec270, %wide.trip.count206
  br label %iter.check

for.cond4.preheader.lr.ph.split.split.us.split.us: ; preds = %for.cond4.preheader.lr.ph.split.split.us
  br i1 %cmp2693, label %for.cond4.preheader.us170.us.us.preheader, label %for.cond4.preheader.us170.us.preheader

for.cond4.preheader.us170.us.preheader:           ; preds = %for.cond4.preheader.lr.ph.split.split.us.split.us
  %6 = sext i32 %add3 to i64
  %7 = add nuw i32 %div, 1
  %wide.trip.count225 = zext i32 %7 to i64
  %wide.trip.count218 = zext i32 %add3 to i64
  %min.iters.check275 = icmp ult i32 %add3, 16
  %min.iters.check278 = icmp ult i32 %add3, 128
  %n.vec282 = and i64 %wide.trip.count218, 4294967168
  %cmp.n283 = icmp eq i64 %n.vec282, %wide.trip.count218
  %n.vec.remaining291 = and i64 %wide.trip.count218, 112
  %min.epilog.iters.check292 = icmp eq i64 %n.vec.remaining291, 0
  %n.vec295 = and i64 %wide.trip.count218, 4294967280
  %cmp.n297 = icmp eq i64 %n.vec295, %wide.trip.count218
  br label %iter.check277

for.cond4.preheader.us170.us.us.preheader:        ; preds = %for.cond4.preheader.lr.ph.split.split.us.split.us
  %8 = sext i32 %input_width to i64
  %9 = zext i32 %depth to i64
  %10 = zext i32 %filter_width to i64
  %11 = sext i32 %stride to i64
  %12 = sext i32 %add3 to i64
  %13 = add nuw i32 %div, 1
  %wide.trip.count260 = zext i32 %13 to i64
  %wide.trip.count253 = zext i32 %add3 to i64
  %wide.trip.count246 = zext i32 %filter_height to i64
  %wide.trip.count240 = zext i32 %filter_width to i64
  %wide.trip.count231 = zext i32 %depth to i64
  %min.iters.check302 = icmp ult i32 %depth, 16
  %min.iters.check305 = icmp ult i32 %depth, 128
  %n.vec309 = and i64 %wide.trip.count231, 4294967168
  %cmp.n310 = icmp eq i64 %n.vec309, %wide.trip.count231
  %n.vec.remaining330 = and i64 %wide.trip.count231, 112
  %min.epilog.iters.check331 = icmp eq i64 %n.vec.remaining330, 0
  %n.vec334 = and i64 %wide.trip.count231, 4294967280
  %cmp.n336 = icmp eq i64 %n.vec334, %wide.trip.count231
  br label %for.cond4.preheader.us170.us.us

for.cond4.preheader.us170.us.us:                  ; preds = %for.cond4.preheader.us170.us.us.preheader, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us
  %indvars.iv255 = phi i64 [ 0, %for.cond4.preheader.us170.us.us.preheader ], [ %indvars.iv.next256, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us ]
  %14 = mul nsw i64 %indvars.iv255, %11
  %15 = mul nsw i64 %indvars.iv255, %12
  br label %for.cond8.preheader.us.us.us.us.us.us

for.cond8.preheader.us.us.us.us.us.us:            ; preds = %for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us, %for.cond4.preheader.us170.us.us
  %indvars.iv248 = phi i64 [ %indvars.iv.next249, %for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us ], [ 0, %for.cond4.preheader.us170.us.us ]
  %16 = mul nsw i64 %indvars.iv248, %11
  br label %for.cond12.preheader.us.us.us.us.us.us.us.us

for.cond12.preheader.us.us.us.us.us.us.us.us:     ; preds = %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us, %for.cond8.preheader.us.us.us.us.us.us
  %indvars.iv242 = phi i64 [ %indvars.iv.next243, %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us ], [ 0, %for.cond8.preheader.us.us.us.us.us.us ]
  %sum_block.0113.us.us.us.us.us.us.us.us = phi i8 [ %.us-phi.us.us.us.us.us.us.us.us, %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us ], [ 0, %for.cond8.preheader.us.us.us.us.us.us ]
  %17 = add nsw i64 %indvars.iv242, %14
  %.fr = freeze i64 %17
  %18 = trunc i64 %.fr to i32
  %cmp19.us.us.us.us.us.us.us.us = icmp sgt i32 %18, -1
  %19 = mul nsw i64 %indvars.iv242, %10
  %cmp20.us.us.us.us.us.us.us.us = icmp slt i32 %18, %input_height
  %or.cond = and i1 %cmp19.us.us.us.us.us.us.us.us, %cmp20.us.us.us.us.us.us.us.us
  br i1 %or.cond, label %for.body15.us.us102.us.us.us.us.us.us.us.us.preheader, label %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us

for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us: ; preds = %if.end.us.us105.us.us.us.us.us.us.us.us, %for.cond12.preheader.us.us.us.us.us.us.us.us
  %.us-phi.us.us.us.us.us.us.us.us = phi i8 [ %sum_block.0113.us.us.us.us.us.us.us.us, %for.cond12.preheader.us.us.us.us.us.us.us.us ], [ %sum_block.3.us.us106.us.us.us.us.us.us.us.us, %if.end.us.us105.us.us.us.us.us.us.us.us ]
  %indvars.iv.next243 = add nuw nsw i64 %indvars.iv242, 1
  %exitcond247.not = icmp eq i64 %indvars.iv.next243, %wide.trip.count246
  br i1 %exitcond247.not, label %for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us, label %for.cond12.preheader.us.us.us.us.us.us.us.us, !llvm.loop !5

for.body15.us.us102.us.us.us.us.us.us.us.us.preheader: ; preds = %for.cond12.preheader.us.us.us.us.us.us.us.us
  %mul29.us.us.us.us.us.us.us.us = mul nsw i32 %18, %input_width
  %20 = sext i32 %mul29.us.us.us.us.us.us.us.us to i64
  br label %for.body15.us.us102.us.us.us.us.us.us.us.us

for.body15.us.us102.us.us.us.us.us.us.us.us:      ; preds = %for.body15.us.us102.us.us.us.us.us.us.us.us.preheader, %if.end.us.us105.us.us.us.us.us.us.us.us
  %indvars.iv233 = phi i64 [ 0, %for.body15.us.us102.us.us.us.us.us.us.us.us.preheader ], [ %indvars.iv.next234, %if.end.us.us105.us.us.us.us.us.us.us.us ]
  %sum_block.199.us.us103.us.us.us.us.us.us.us.us = phi i8 [ %sum_block.0113.us.us.us.us.us.us.us.us, %for.body15.us.us102.us.us.us.us.us.us.us.us.preheader ], [ %sum_block.3.us.us106.us.us.us.us.us.us.us.us, %if.end.us.us105.us.us.us.us.us.us.us.us ]
  %21 = add nsw i64 %indvars.iv233, %16
  %cmp22.us.us.us.us.us.us.us.us.us.us = icmp sgt i64 %21, -1
  %cmp24.us.us.us.us.us.us.us.us.us.us = icmp slt i64 %21, %8
  %or.cond92.us.us.us.us.us.us.us.us.us.us = and i1 %cmp22.us.us.us.us.us.us.us.us.us.us, %cmp24.us.us.us.us.us.us.us.us.us.us
  br i1 %or.cond92.us.us.us.us.us.us.us.us.us.us, label %iter.check304, label %if.end.us.us105.us.us.us.us.us.us.us.us

if.end.us.us105.us.us.us.us.us.us.us.us:          ; preds = %for.body28.us.us.us.us.us.us.us.us.us.us, %middle.block301, %vec.epilog.middle.block326, %for.body15.us.us102.us.us.us.us.us.us.us.us
  %sum_block.3.us.us106.us.us.us.us.us.us.us.us = phi i8 [ %sum_block.199.us.us103.us.us.us.us.us.us.us.us, %for.body15.us.us102.us.us.us.us.us.us.us.us ], [ %50, %middle.block301 ], [ %59, %vec.epilog.middle.block326 ], [ %add42.us.us.us.us.us.us.us.us.us.us, %for.body28.us.us.us.us.us.us.us.us.us.us ]
  %indvars.iv.next234 = add nuw nsw i64 %indvars.iv233, 1
  %exitcond241.not = icmp eq i64 %indvars.iv.next234, %wide.trip.count240
  br i1 %exitcond241.not, label %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us, label %for.body15.us.us102.us.us.us.us.us.us.us.us, !llvm.loop !7

for.body28.us.us.us.us.us.us.us.us.us.us:         ; preds = %for.body28.us.us.us.us.us.us.us.us.us.us.preheader, %for.body28.us.us.us.us.us.us.us.us.us.us
  %indvars.iv227 = phi i64 [ %indvars.iv.next228, %for.body28.us.us.us.us.us.us.us.us.us.us ], [ %indvars.iv227.ph, %for.body28.us.us.us.us.us.us.us.us.us.us.preheader ]
  %sum_block.294.us.us.us.us.us.us.us.us.us.us = phi i8 [ %add42.us.us.us.us.us.us.us.us.us.us, %for.body28.us.us.us.us.us.us.us.us.us.us ], [ %sum_block.294.us.us.us.us.us.us.us.us.us.us.ph, %for.body28.us.us.us.us.us.us.us.us.us.us.preheader ]
  %22 = add nsw i64 %indvars.iv227, %27
  %arrayidx.us.us.us.us.us.us.us.us.us.us = getelementptr inbounds i8, ptr %input, i64 %22
  %23 = load i8, ptr %arrayidx.us.us.us.us.us.us.us.us.us.us, align 1, !tbaa !8
  %24 = add nuw nsw i64 %indvars.iv227, %29
  %arrayidx38.us.us.us.us.us.us.us.us.us.us = getelementptr inbounds i8, ptr %filter, i64 %24
  %25 = load i8, ptr %arrayidx38.us.us.us.us.us.us.us.us.us.us, align 1, !tbaa !8
  %mul40.us.us.us.us.us.us.us.us.us.us = mul i8 %25, %23
  %add42.us.us.us.us.us.us.us.us.us.us = add i8 %mul40.us.us.us.us.us.us.us.us.us.us, %sum_block.294.us.us.us.us.us.us.us.us.us.us
  %indvars.iv.next228 = add nuw nsw i64 %indvars.iv227, 1
  %exitcond232.not = icmp eq i64 %indvars.iv.next228, %wide.trip.count231
  br i1 %exitcond232.not, label %if.end.us.us105.us.us.us.us.us.us.us.us, label %for.body28.us.us.us.us.us.us.us.us.us.us, !llvm.loop !11

iter.check304:                                    ; preds = %for.body15.us.us102.us.us.us.us.us.us.us.us
  %26 = add nsw i64 %21, %20
  %27 = mul nsw i64 %26, %9
  %28 = add nuw nsw i64 %indvars.iv233, %19
  %29 = mul nsw i64 %28, %9
  br i1 %min.iters.check302, label %for.body28.us.us.us.us.us.us.us.us.us.us.preheader, label %vector.main.loop.iter.check306

vector.main.loop.iter.check306:                   ; preds = %iter.check304
  br i1 %min.iters.check305, label %vec.epilog.ph329, label %vector.ph307

vector.ph307:                                     ; preds = %vector.main.loop.iter.check306
  %30 = insertelement <32 x i8> <i8 poison, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %sum_block.199.us.us103.us.us.us.us.us.us.us.us, i64 0
  br label %vector.body311

vector.body311:                                   ; preds = %vector.body311, %vector.ph307
  %index312 = phi i64 [ 0, %vector.ph307 ], [ %index.next323, %vector.body311 ]
  %vec.phi = phi <32 x i8> [ %30, %vector.ph307 ], [ %45, %vector.body311 ]
  %vec.phi313 = phi <32 x i8> [ zeroinitializer, %vector.ph307 ], [ %46, %vector.body311 ]
  %vec.phi314 = phi <32 x i8> [ zeroinitializer, %vector.ph307 ], [ %47, %vector.body311 ]
  %vec.phi315 = phi <32 x i8> [ zeroinitializer, %vector.ph307 ], [ %48, %vector.body311 ]
  %31 = add nsw i64 %index312, %27
  %32 = getelementptr inbounds i8, ptr %input, i64 %31
  %wide.load = load <32 x i8>, ptr %32, align 1, !tbaa !8
  %33 = getelementptr inbounds i8, ptr %32, i64 32
  %wide.load316 = load <32 x i8>, ptr %33, align 1, !tbaa !8
  %34 = getelementptr inbounds i8, ptr %32, i64 64
  %wide.load317 = load <32 x i8>, ptr %34, align 1, !tbaa !8
  %35 = getelementptr inbounds i8, ptr %32, i64 96
  %wide.load318 = load <32 x i8>, ptr %35, align 1, !tbaa !8
  %36 = add nuw nsw i64 %index312, %29
  %37 = getelementptr inbounds i8, ptr %filter, i64 %36
  %wide.load319 = load <32 x i8>, ptr %37, align 1, !tbaa !8
  %38 = getelementptr inbounds i8, ptr %37, i64 32
  %wide.load320 = load <32 x i8>, ptr %38, align 1, !tbaa !8
  %39 = getelementptr inbounds i8, ptr %37, i64 64
  %wide.load321 = load <32 x i8>, ptr %39, align 1, !tbaa !8
  %40 = getelementptr inbounds i8, ptr %37, i64 96
  %wide.load322 = load <32 x i8>, ptr %40, align 1, !tbaa !8
  %41 = mul <32 x i8> %wide.load319, %wide.load
  %42 = mul <32 x i8> %wide.load320, %wide.load316
  %43 = mul <32 x i8> %wide.load321, %wide.load317
  %44 = mul <32 x i8> %wide.load322, %wide.load318
  %45 = add <32 x i8> %41, %vec.phi
  %46 = add <32 x i8> %42, %vec.phi313
  %47 = add <32 x i8> %43, %vec.phi314
  %48 = add <32 x i8> %44, %vec.phi315
  %index.next323 = add nuw i64 %index312, 128
  %49 = icmp eq i64 %index.next323, %n.vec309
  br i1 %49, label %middle.block301, label %vector.body311, !llvm.loop !14

middle.block301:                                  ; preds = %vector.body311
  %bin.rdx = add <32 x i8> %46, %45
  %bin.rdx324 = add <32 x i8> %47, %bin.rdx
  %bin.rdx325 = add <32 x i8> %48, %bin.rdx324
  %50 = tail call i8 @llvm.vector.reduce.add.v32i8(<32 x i8> %bin.rdx325)
  br i1 %cmp.n310, label %if.end.us.us105.us.us.us.us.us.us.us.us, label %vec.epilog.iter.check328

vec.epilog.iter.check328:                         ; preds = %middle.block301
  br i1 %min.epilog.iters.check331, label %for.body28.us.us.us.us.us.us.us.us.us.us.preheader, label %vec.epilog.ph329

vec.epilog.ph329:                                 ; preds = %vector.main.loop.iter.check306, %vec.epilog.iter.check328
  %bc.merge.rdx = phi i8 [ %sum_block.199.us.us103.us.us.us.us.us.us.us.us, %vector.main.loop.iter.check306 ], [ %50, %vec.epilog.iter.check328 ]
  %vec.epilog.resume.val332 = phi i64 [ 0, %vector.main.loop.iter.check306 ], [ %n.vec309, %vec.epilog.iter.check328 ]
  %51 = insertelement <16 x i8> <i8 poison, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %bc.merge.rdx, i64 0
  br label %vec.epilog.vector.body337

vec.epilog.vector.body337:                        ; preds = %vec.epilog.vector.body337, %vec.epilog.ph329
  %index338 = phi i64 [ %vec.epilog.resume.val332, %vec.epilog.ph329 ], [ %index.next342, %vec.epilog.vector.body337 ]
  %vec.phi339 = phi <16 x i8> [ %51, %vec.epilog.ph329 ], [ %57, %vec.epilog.vector.body337 ]
  %52 = add nsw i64 %index338, %27
  %53 = getelementptr inbounds i8, ptr %input, i64 %52
  %wide.load340 = load <16 x i8>, ptr %53, align 1, !tbaa !8
  %54 = add nuw nsw i64 %index338, %29
  %55 = getelementptr inbounds i8, ptr %filter, i64 %54
  %wide.load341 = load <16 x i8>, ptr %55, align 1, !tbaa !8
  %56 = mul <16 x i8> %wide.load341, %wide.load340
  %57 = add <16 x i8> %56, %vec.phi339
  %index.next342 = add nuw i64 %index338, 16
  %58 = icmp eq i64 %index.next342, %n.vec334
  br i1 %58, label %vec.epilog.middle.block326, label %vec.epilog.vector.body337, !llvm.loop !15

vec.epilog.middle.block326:                       ; preds = %vec.epilog.vector.body337
  %59 = tail call i8 @llvm.vector.reduce.add.v16i8(<16 x i8> %57)
  br i1 %cmp.n336, label %if.end.us.us105.us.us.us.us.us.us.us.us, label %for.body28.us.us.us.us.us.us.us.us.us.us.preheader

for.body28.us.us.us.us.us.us.us.us.us.us.preheader: ; preds = %iter.check304, %vec.epilog.iter.check328, %vec.epilog.middle.block326
  %indvars.iv227.ph = phi i64 [ 0, %iter.check304 ], [ %n.vec309, %vec.epilog.iter.check328 ], [ %n.vec334, %vec.epilog.middle.block326 ]
  %sum_block.294.us.us.us.us.us.us.us.us.us.us.ph = phi i8 [ %sum_block.199.us.us103.us.us.us.us.us.us.us.us, %iter.check304 ], [ %50, %vec.epilog.iter.check328 ], [ %59, %vec.epilog.middle.block326 ]
  br label %for.body28.us.us.us.us.us.us.us.us.us.us

for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us: ; preds = %for.cond12.for.cond.cleanup14_crit_edge.us.us.us.us.us.us.us.us
  %60 = add nsw i64 %indvars.iv248, %15
  %arrayidx53.us.us.us.us.us.us = getelementptr inbounds i8, ptr %output, i64 %60
  store i8 %.us-phi.us.us.us.us.us.us.us.us, ptr %arrayidx53.us.us.us.us.us.us, align 1, !tbaa !8
  %indvars.iv.next249 = add nuw nsw i64 %indvars.iv248, 1
  %exitcond254.not = icmp eq i64 %indvars.iv.next249, %wide.trip.count253
  br i1 %exitcond254.not, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us, label %for.cond8.preheader.us.us.us.us.us.us, !llvm.loop !16

for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us: ; preds = %for.cond8.for.cond.cleanup10_crit_edge.split.us.split.us.us.us.us.us.us.us
  %indvars.iv.next256 = add nuw nsw i64 %indvars.iv255, 1
  %exitcond261.not = icmp eq i64 %indvars.iv.next256, %wide.trip.count260
  br i1 %exitcond261.not, label %for.cond.cleanup, label %for.cond4.preheader.us170.us.us, !llvm.loop !17

iter.check277:                                    ; preds = %for.cond4.preheader.us170.us.preheader, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us
  %indvars.iv220 = phi i64 [ 0, %for.cond4.preheader.us170.us.preheader ], [ %indvars.iv.next221, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us ]
  %61 = mul nsw i64 %indvars.iv220, %6
  br i1 %min.iters.check275, label %for.cond8.preheader.us.us.us183.us.preheader, label %vector.main.loop.iter.check279

vector.main.loop.iter.check279:                   ; preds = %iter.check277
  br i1 %min.iters.check278, label %vec.epilog.ph290, label %vector.body284

vector.body284:                                   ; preds = %vector.main.loop.iter.check279, %vector.body284
  %index285 = phi i64 [ %index.next286, %vector.body284 ], [ 0, %vector.main.loop.iter.check279 ]
  %62 = add nsw i64 %index285, %61
  %63 = getelementptr inbounds i8, ptr %output, i64 %62
  store <32 x i8> zeroinitializer, ptr %63, align 1, !tbaa !8
  %64 = getelementptr inbounds i8, ptr %63, i64 32
  store <32 x i8> zeroinitializer, ptr %64, align 1, !tbaa !8
  %65 = getelementptr inbounds i8, ptr %63, i64 64
  store <32 x i8> zeroinitializer, ptr %65, align 1, !tbaa !8
  %66 = getelementptr inbounds i8, ptr %63, i64 96
  store <32 x i8> zeroinitializer, ptr %66, align 1, !tbaa !8
  %index.next286 = add nuw i64 %index285, 128
  %67 = icmp eq i64 %index.next286, %n.vec282
  br i1 %67, label %middle.block274, label %vector.body284, !llvm.loop !18

middle.block274:                                  ; preds = %vector.body284
  br i1 %cmp.n283, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us, label %vec.epilog.iter.check289

vec.epilog.iter.check289:                         ; preds = %middle.block274
  br i1 %min.epilog.iters.check292, label %for.cond8.preheader.us.us.us183.us.preheader, label %vec.epilog.ph290

vec.epilog.ph290:                                 ; preds = %vector.main.loop.iter.check279, %vec.epilog.iter.check289
  %vec.epilog.resume.val293 = phi i64 [ %n.vec282, %vec.epilog.iter.check289 ], [ 0, %vector.main.loop.iter.check279 ]
  br label %vec.epilog.vector.body298

vec.epilog.vector.body298:                        ; preds = %vec.epilog.vector.body298, %vec.epilog.ph290
  %index299 = phi i64 [ %vec.epilog.resume.val293, %vec.epilog.ph290 ], [ %index.next300, %vec.epilog.vector.body298 ]
  %68 = add nsw i64 %index299, %61
  %69 = getelementptr inbounds i8, ptr %output, i64 %68
  store <16 x i8> zeroinitializer, ptr %69, align 1, !tbaa !8
  %index.next300 = add nuw i64 %index299, 16
  %70 = icmp eq i64 %index.next300, %n.vec295
  br i1 %70, label %vec.epilog.middle.block287, label %vec.epilog.vector.body298, !llvm.loop !19

vec.epilog.middle.block287:                       ; preds = %vec.epilog.vector.body298
  br i1 %cmp.n297, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us, label %for.cond8.preheader.us.us.us183.us.preheader

for.cond8.preheader.us.us.us183.us.preheader:     ; preds = %iter.check277, %vec.epilog.iter.check289, %vec.epilog.middle.block287
  %indvars.iv214.ph = phi i64 [ 0, %iter.check277 ], [ %n.vec282, %vec.epilog.iter.check289 ], [ %n.vec295, %vec.epilog.middle.block287 ]
  br label %for.cond8.preheader.us.us.us183.us

for.cond8.preheader.us.us.us183.us:               ; preds = %for.cond8.preheader.us.us.us183.us.preheader, %for.cond8.preheader.us.us.us183.us
  %indvars.iv214 = phi i64 [ %indvars.iv.next215, %for.cond8.preheader.us.us.us183.us ], [ %indvars.iv214.ph, %for.cond8.preheader.us.us.us183.us.preheader ]
  %71 = add nsw i64 %indvars.iv214, %61
  %arrayidx53.us.us.us188.us = getelementptr inbounds i8, ptr %output, i64 %71
  store i8 0, ptr %arrayidx53.us.us.us188.us, align 1, !tbaa !8
  %indvars.iv.next215 = add nuw nsw i64 %indvars.iv214, 1
  %exitcond219.not = icmp eq i64 %indvars.iv.next215, %wide.trip.count218
  br i1 %exitcond219.not, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us, label %for.cond8.preheader.us.us.us183.us, !llvm.loop !20

for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us: ; preds = %for.cond8.preheader.us.us.us183.us, %vec.epilog.middle.block287, %middle.block274
  %indvars.iv.next221 = add nuw nsw i64 %indvars.iv220, 1
  %exitcond226.not = icmp eq i64 %indvars.iv.next221, %wide.trip.count225
  br i1 %exitcond226.not, label %for.cond.cleanup, label %iter.check277, !llvm.loop !17

iter.check:                                       ; preds = %for.cond4.preheader.us170.preheader, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181
  %indvars.iv208 = phi i64 [ 0, %for.cond4.preheader.us170.preheader ], [ %indvars.iv.next209, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181 ]
  %72 = mul nsw i64 %indvars.iv208, %4
  br i1 %min.iters.check, label %for.cond8.preheader.us.us174.preheader, label %vector.main.loop.iter.check

vector.main.loop.iter.check:                      ; preds = %iter.check
  br i1 %min.iters.check268, label %vec.epilog.ph, label %vector.body

vector.body:                                      ; preds = %vector.main.loop.iter.check, %vector.body
  %index = phi i64 [ %index.next, %vector.body ], [ 0, %vector.main.loop.iter.check ]
  %73 = add nsw i64 %index, %72
  %74 = getelementptr inbounds i8, ptr %output, i64 %73
  store <32 x i8> zeroinitializer, ptr %74, align 1, !tbaa !8
  %75 = getelementptr inbounds i8, ptr %74, i64 32
  store <32 x i8> zeroinitializer, ptr %75, align 1, !tbaa !8
  %76 = getelementptr inbounds i8, ptr %74, i64 64
  store <32 x i8> zeroinitializer, ptr %76, align 1, !tbaa !8
  %77 = getelementptr inbounds i8, ptr %74, i64 96
  store <32 x i8> zeroinitializer, ptr %77, align 1, !tbaa !8
  %index.next = add nuw i64 %index, 128
  %78 = icmp eq i64 %index.next, %n.vec
  br i1 %78, label %middle.block, label %vector.body, !llvm.loop !21

middle.block:                                     ; preds = %vector.body
  br i1 %cmp.n, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181, label %vec.epilog.iter.check

vec.epilog.iter.check:                            ; preds = %middle.block
  br i1 %min.epilog.iters.check, label %for.cond8.preheader.us.us174.preheader, label %vec.epilog.ph

vec.epilog.ph:                                    ; preds = %vector.main.loop.iter.check, %vec.epilog.iter.check
  %vec.epilog.resume.val = phi i64 [ %n.vec, %vec.epilog.iter.check ], [ 0, %vector.main.loop.iter.check ]
  br label %vec.epilog.vector.body

vec.epilog.vector.body:                           ; preds = %vec.epilog.vector.body, %vec.epilog.ph
  %index272 = phi i64 [ %vec.epilog.resume.val, %vec.epilog.ph ], [ %index.next273, %vec.epilog.vector.body ]
  %79 = add nsw i64 %index272, %72
  %80 = getelementptr inbounds i8, ptr %output, i64 %79
  store <16 x i8> zeroinitializer, ptr %80, align 1, !tbaa !8
  %index.next273 = add nuw i64 %index272, 16
  %81 = icmp eq i64 %index.next273, %n.vec270
  br i1 %81, label %vec.epilog.middle.block, label %vec.epilog.vector.body, !llvm.loop !22

vec.epilog.middle.block:                          ; preds = %vec.epilog.vector.body
  br i1 %cmp.n271, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181, label %for.cond8.preheader.us.us174.preheader

for.cond8.preheader.us.us174.preheader:           ; preds = %iter.check, %vec.epilog.iter.check, %vec.epilog.middle.block
  %indvars.iv.ph = phi i64 [ 0, %iter.check ], [ %n.vec, %vec.epilog.iter.check ], [ %n.vec270, %vec.epilog.middle.block ]
  br label %for.cond8.preheader.us.us174

for.cond8.preheader.us.us174:                     ; preds = %for.cond8.preheader.us.us174.preheader, %for.cond8.preheader.us.us174
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.cond8.preheader.us.us174 ], [ %indvars.iv.ph, %for.cond8.preheader.us.us174.preheader ]
  %82 = add nsw i64 %indvars.iv, %72
  %arrayidx53.us.us178 = getelementptr inbounds i8, ptr %output, i64 %82
  store i8 0, ptr %arrayidx53.us.us178, align 1, !tbaa !8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond207.not = icmp eq i64 %indvars.iv.next, %wide.trip.count206
  br i1 %exitcond207.not, label %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181, label %for.cond8.preheader.us.us174, !llvm.loop !23

for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181: ; preds = %for.cond8.preheader.us.us174, %vec.epilog.middle.block, %middle.block
  %indvars.iv.next209 = add nuw nsw i64 %indvars.iv208, 1
  %exitcond213.not = icmp eq i64 %indvars.iv.next209, %wide.trip.count212
  br i1 %exitcond213.not, label %for.cond.cleanup, label %iter.check, !llvm.loop !17

for.cond4.preheader:                              ; preds = %for.cond4.preheader, %for.cond4.preheader.preheader.new
  %indvar = phi i64 [ 0, %for.cond4.preheader.preheader.new ], [ %indvar.next.7, %for.cond4.preheader ]
  %niter = phi i64 [ 0, %for.cond4.preheader.preheader.new ], [ %niter.next.7, %for.cond4.preheader ]
  %83 = mul nsw i64 %indvar, %0
  %scevgep = getelementptr i8, ptr %output, i64 %83
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next = or i64 %indvar, 1
  %84 = mul nsw i64 %indvar.next, %0
  %scevgep.1 = getelementptr i8, ptr %output, i64 %84
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.1, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.1 = or i64 %indvar, 2
  %85 = mul nsw i64 %indvar.next.1, %0
  %scevgep.2 = getelementptr i8, ptr %output, i64 %85
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.2, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.2 = or i64 %indvar, 3
  %86 = mul nsw i64 %indvar.next.2, %0
  %scevgep.3 = getelementptr i8, ptr %output, i64 %86
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.3, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.3 = or i64 %indvar, 4
  %87 = mul nsw i64 %indvar.next.3, %0
  %scevgep.4 = getelementptr i8, ptr %output, i64 %87
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.4, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.4 = or i64 %indvar, 5
  %88 = mul nsw i64 %indvar.next.4, %0
  %scevgep.5 = getelementptr i8, ptr %output, i64 %88
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.5, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.5 = or i64 %indvar, 6
  %89 = mul nsw i64 %indvar.next.5, %0
  %scevgep.6 = getelementptr i8, ptr %output, i64 %89
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.6, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.6 = or i64 %indvar, 7
  %90 = mul nsw i64 %indvar.next.6, %0
  %scevgep.7 = getelementptr i8, ptr %output, i64 %90
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.7, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.7 = add nuw nsw i64 %indvar, 8
  %niter.next.7 = add i64 %niter, 8
  %niter.ncmp.7 = icmp eq i64 %niter.next.7, %unroll_iter
  br i1 %niter.ncmp.7, label %for.cond.cleanup.loopexit350.unr-lcssa, label %for.cond4.preheader, !llvm.loop !17

for.cond.cleanup.loopexit350.unr-lcssa:           ; preds = %for.cond4.preheader, %for.cond4.preheader.preheader
  %indvar.unr = phi i64 [ 0, %for.cond4.preheader.preheader ], [ %indvar.next.7, %for.cond4.preheader ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.cond4.preheader.epil

for.cond4.preheader.epil:                         ; preds = %for.cond.cleanup.loopexit350.unr-lcssa, %for.cond4.preheader.epil
  %indvar.epil = phi i64 [ %indvar.next.epil, %for.cond4.preheader.epil ], [ %indvar.unr, %for.cond.cleanup.loopexit350.unr-lcssa ]
  %epil.iter = phi i64 [ %epil.iter.next, %for.cond4.preheader.epil ], [ 0, %for.cond.cleanup.loopexit350.unr-lcssa ]
  %91 = mul nsw i64 %indvar.epil, %0
  %scevgep.epil = getelementptr i8, ptr %output, i64 %91
  tail call void @llvm.memset.p0.i64(ptr align 1 %scevgep.epil, i8 0, i64 %1, i1 false), !tbaa !8
  %indvar.next.epil = add nuw nsw i64 %indvar.epil, 1
  %epil.iter.next = add i64 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i64 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.cond4.preheader.epil, !llvm.loop !24

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit350.unr-lcssa, %for.cond4.preheader.epil, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us181, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us191.us, %for.cond4.for.cond.cleanup6_crit_edge.split.us.split.us.split.us.us.us.us, %for.cond4.preheader.lr.ph, %entry
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.vector.reduce.add.v32i8(<32 x i8>) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.vector.reduce.add.v16i8(<16 x i8>) #2

attributes #0 = { nofree nosync nounwind memory(argmem: readwrite) uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="alderlake" "target-features"="+64bit,+adx,+aes,+avx,+avx2,+avxvnni,+bmi,+bmi2,+clflushopt,+clwb,+cmov,+crc32,+cx16,+cx8,+f16c,+fma,+fsgsbase,+fxsr,+gfni,+hreset,+invpcid,+kl,+lzcnt,+mmx,+movbe,+movdir64b,+movdiri,+pclmul,+pconfig,+pku,+popcnt,+prfchw,+ptwrite,+rdpid,+rdrnd,+rdseed,+sahf,+serialize,+sha,+shstk,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+vaes,+vpclmulqdq,+waitpkg,+widekl,+x87,+xsave,+xsavec,+xsaveopt,+xsaves,-amx-bf16,-amx-complex,-amx-fp16,-amx-int8,-amx-tile,-avx512bf16,-avx512bitalg,-avx512bw,-avx512cd,-avx512dq,-avx512er,-avx512f,-avx512fp16,-avx512ifma,-avx512pf,-avx512vbmi,-avx512vbmi2,-avx512vl,-avx512vnni,-avx512vp2intersect,-avx512vpopcntdq,-avxifma,-avxneconvert,-avxvnniint16,-avxvnniint8,-cldemote,-clzero,-cmpccxadd,-enqcmd,-fma4,-lwp,-mwaitx,-prefetchi,-prefetchwt1,-raoint,-rdpru,-rtm,-sgx,-sha512,-sm3,-sm4,-sse4a,-tbm,-tsxldtrk,-uintr,-wbnoinvd,-xop" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{!"clang version 17.0.1 (https://github.com/llvm/llvm-project.git e19b7dc36bc047b9eb72078d034596be766da350)"}
!5 = distinct !{!5, !6}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = distinct !{!11, !6, !12, !13}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = distinct !{!14, !6, !13, !12}
!15 = distinct !{!15, !6, !13, !12}
!16 = distinct !{!16, !6}
!17 = distinct !{!17, !6}
!18 = distinct !{!18, !6, !13, !12}
!19 = distinct !{!19, !6, !13, !12}
!20 = distinct !{!20, !6, !12, !13}
!21 = distinct !{!21, !6, !13, !12}
!22 = distinct !{!22, !6, !13, !12}
!23 = distinct !{!23, !6, !12, !13}
!24 = distinct !{!24, !25}
!25 = !{!"llvm.loop.unroll.disable"}
