/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipeline;

import edu.wpi.first.apriltag.AprilTagDetection;
import edu.wpi.first.apriltag.AprilTagDetector;
import edu.wpi.first.apriltag.AprilTagPoseEstimate;
import edu.wpi.first.apriltag.AprilTagPoseEstimator.Config;
import edu.wpi.first.math.geometry.CoordinateSystem;
import edu.wpi.first.math.geometry.Pose3d;
import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;
import edu.wpi.first.math.util.Units;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.opencv.core.Mat;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.Contour;
import org.photonvision.vision.opencv.DualOffsetValues;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe;
import org.photonvision.vision.pipe.impl.AlgaeDetectionPipe.AlgaeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.Collect2dTargetsPipe;
import org.photonvision.vision.pipe.impl.Draw2dCrosshairPipe;
import org.photonvision.vision.pipe.impl.Draw2dTargetsPipe;
import org.photonvision.vision.pipe.impl.Draw3dTargetsPipe;
import org.photonvision.vision.pipe.impl.EdgeMaskPipe;
import org.photonvision.vision.pipe.impl.FindContoursPipe;
import org.photonvision.vision.pipe.impl.GrayscalePipe;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.pipe.impl.PaddingPipe;
import org.photonvision.vision.pipe.impl.UnPadPipe;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.PotentialTarget;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;

public class AprilTagAlgaePipeline
        extends CVPipeline<CVPipelineResult, AprilTagAlgaePipelineSettings> {
    // APRIL TAG
    private final GrayscalePipe grayscalePipe = new GrayscalePipe();
    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe =
            new AprilTagPoseEstimatorPipe();
    private final MultiTargetPNPPipe multiTagPNPPipe = new MultiTargetPNPPipe();

    // ALGAE
    private final PaddingPipe paddedPipe = new PaddingPipe();
    private final HSVPipe hsvPipe = new HSVPipe();
    private final EdgeMaskPipe maskPipe = new EdgeMaskPipe();
    private final FindContoursPipe findContoursPipe = new FindContoursPipe();
    private final UnPadPipe unPadPipe = new UnPadPipe();
    private final AlgaeDetectionPipe algaeDetectionPipe = new AlgaeDetectionPipe();
    private final Collect2dTargetsPipe collect2dTargetsPipe = new Collect2dTargetsPipe();
    private final Draw2dCrosshairPipe draw2dCrosshairPipe = new Draw2dCrosshairPipe();
    private final Draw2dTargetsPipe draw2DTargetsPipe = new Draw2dTargetsPipe();
    private final Draw3dTargetsPipe draw3dTargetsPipe = new Draw3dTargetsPipe();

    // COMMON
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.NONE;

    public AprilTagAlgaePipeline() {
        super(PROCESSING_TYPE);
        settings = new AprilTagAlgaePipelineSettings();
    }

    public AprilTagAlgaePipeline(AprilTagAlgaePipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
    }

    @Override
    protected void setPipeParamsImpl() {
        // APRIL TAG
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        var grayscaleParams = new GrayscalePipe.GrayscaleParams();
        grayscalePipe.setParams(grayscaleParams);

        // for now, hard code tag width based on enum value
        // 2023/other: best guess is 6in
        double tagWidth = Units.inchesToMeters(6);
        TargetModel tagModel = TargetModel.kAprilTag16h5;
        if (settings.tagFamily == AprilTagFamily.kTag36h11) {
            // 2024 tag, 6.5in
            tagWidth = Units.inchesToMeters(6.5);
            tagModel = TargetModel.kAprilTag36h11;
        }

        var config = new AprilTagDetector.Config();
        config.numThreads = settings.threads;
        config.refineEdges = settings.refineEdges;
        config.quadSigma = (float) settings.blur;
        config.quadDecimate = settings.decimate;

        var quadParams = new AprilTagDetector.QuadThresholdParameters();
        // 5 was the default minClusterPixels in WPILib prior to 2025
        // increasing it causes detection problems when decimate > 1
        quadParams.minClusterPixels = 5;
        // these are the same as the values in WPILib 2025
        // setting them here to prevent upstream changes from changing behavior of the
        // detector
        quadParams.maxNumMaxima = 10;
        quadParams.criticalAngle = 45 * Math.PI / 180.0;
        quadParams.maxLineFitMSE = 10.0f;
        quadParams.minWhiteBlackDiff = 5;
        quadParams.deglitch = false;

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(settings.tagFamily, config, quadParams));

        if (frameStaticProperties.cameraCalibration != null) {
            var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
            if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                var cx = cameraMatrix.get(0, 2)[0];
                var cy = cameraMatrix.get(1, 2)[0];
                var fx = cameraMatrix.get(0, 0)[0];
                var fy = cameraMatrix.get(1, 1)[0];

                singleTagPoseEstimatorPipe.setParams(
                        new AprilTagPoseEstimatorPipeParams(
                                new Config(tagWidth, fx, fy, cx, cy),
                                frameStaticProperties.cameraCalibration,
                                settings.numIterations));

                // TODO global state ew
                var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
                multiTagPNPPipe.setParams(
                        new MultiTargetPNPPipeParams(frameStaticProperties.cameraCalibration, atfl, tagModel));
            }
        }

        // ALGAE
        DualOffsetValues dualOffsetValues =
                new DualOffsetValues(
                        settings.offsetDualPointA,
                        settings.offsetDualPointAArea,
                        settings.offsetDualPointB,
                        settings.offsetDualPointBArea);

        PaddingPipe.PaddingParams paddingParams = new PaddingPipe.PaddingParams(settings.padding);
        paddedPipe.setParams(paddingParams);

        HSVPipe.HSVParams hsvParams =
                new HSVPipe.HSVParams(
                        settings.hsvHue, settings.hsvSaturation, settings.hsvValue, settings.hueInverted);
        hsvPipe.setParams(hsvParams);

        EdgeMaskPipe.MaskParams maskParams =
                new EdgeMaskPipe.MaskParams(
                        settings.erosion,
                        settings.initialDilation,
                        settings.edgeThresholds.getFirst(),
                        settings.edgeThresholds.getSecond(),
                        settings.edgeDilation,
                        settings.finalDilation);
        maskPipe.setParams(maskParams);

        FindContoursPipe.FindContoursParams findContoursParams =
                new FindContoursPipe.FindContoursParams();
        findContoursPipe.setParams(findContoursParams);

        UnPadPipe.UnPadParams unPadParams = new UnPadPipe.UnPadParams(settings.padding);
        unPadPipe.setParams(unPadParams);

        AlgaeDetectionPipe.AlgaeDetectionParams algaeDetectionParams =
                new AlgaeDetectionPipe.AlgaeDetectionParams(
                        ((Units.inchesToMeters(16.25)) * 1000),
                        settings.contourArea.getFirst(),
                        settings.contourArea.getSecond(),
                        frameStaticProperties,
                        settings.circularity.getFirst(),
                        settings.circularity.getSecond(),
                        frameStaticProperties.cameraCalibration,
                        settings.padding);
        algaeDetectionPipe.setParams(algaeDetectionParams);
        Collect2dTargetsPipe.Collect2dTargetsParams collect2dTargetsParams =
                new Collect2dTargetsPipe.Collect2dTargetsParams(
                        settings.offsetRobotOffsetMode,
                        settings.offsetSinglePoint,
                        dualOffsetValues,
                        settings.contourTargetOffsetPointEdge,
                        settings.contourTargetOrientation,
                        frameStaticProperties);
        collect2dTargetsPipe.setParams(collect2dTargetsParams);

        Draw2dTargetsPipe.Draw2dTargetsParams draw2DTargetsParams =
                new Draw2dTargetsPipe.Draw2dTargetsParams(
                        settings.outputShouldDraw,
                        settings.outputShowMultipleTargets,
                        settings.streamingFrameDivisor);
        draw2DTargetsParams.showShape = true;
        draw2DTargetsParams.showMaximumBox = false;
        draw2DTargetsParams.showRotatedBox = false;
        draw2DTargetsPipe.setParams(draw2DTargetsParams);

        Draw2dCrosshairPipe.Draw2dCrosshairParams draw2dCrosshairParams =
                new Draw2dCrosshairPipe.Draw2dCrosshairParams(
                        settings.outputShouldDraw,
                        settings.offsetRobotOffsetMode,
                        settings.offsetSinglePoint,
                        dualOffsetValues,
                        frameStaticProperties,
                        settings.streamingFrameDivisor,
                        settings.inputImageRotationMode);
        draw2dCrosshairPipe.setParams(draw2dCrosshairParams);

        var draw3dTargetsParams =
                new Draw3dTargetsPipe.Draw3dContoursParams(
                        settings.outputShouldDraw,
                        frameStaticProperties.cameraCalibration,
                        settings.targetModel,
                        settings.streamingFrameDivisor);
        draw3dTargetsPipe.setParams(draw3dTargetsParams);
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagAlgaePipelineSettings settings) {
        long sumPipeNanosElapsed = 0L;

        // APRIL TAG
        var grayscaleResult = grayscalePipe.run(frame.colorImage.getMat());
        sumPipeNanosElapsed += grayscaleResult.nanosElapsed;

        CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult;
        tagDetectionPipeResult = aprilTagDetectionPipe.run(new CVMat(grayscaleResult.output));
        sumPipeNanosElapsed += tagDetectionPipeResult.nanosElapsed;

        List<AprilTagDetection> detections = tagDetectionPipeResult.output;
        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        for (AprilTagDetection detection : detections) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            usedDetections.add(detection);

            // Populate target list for multitag
            // (TODO: Address circular dependencies. Multitag only requires corners and IDs,
            // this should
            // not be necessary.)
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            null,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, frameStaticProperties));

            targetList.add(target);
        }

        // Do multi-tag pose estimation
        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();
        if (settings.solvePNPEnabled && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        if (settings.solvePNPEnabled) {
            // Clear target list that was used for multitag so we can add target transforms
            targetList.clear();
            // TODO global state again ew
            var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();

            for (AprilTagDetection detection : usedDetections) {
                AprilTagPoseEstimate tagPoseEstimate = null;
                // Do single-tag estimation when "always enabled" or if a tag was not used for
                // multitag
                if (settings.doSingleTargetAlways
                        || !(multiTagResult.isPresent()
                                && multiTagResult.get().fiducialIDsUsed.contains((short) detection.getId()))) {
                    var poseResult = singleTagPoseEstimatorPipe.run(detection);
                    sumPipeNanosElapsed += poseResult.nanosElapsed;
                    tagPoseEstimate = poseResult.output;
                }

                // If single-tag estimation was not done, this is a multi-target tag from the
                // layout
                if (tagPoseEstimate == null && multiTagResult.isPresent()) {
                    // compute this tag's camera-to-tag transform using the multitag result
                    var tagPose = atfl.getTagPose(detection.getId());
                    if (tagPose.isPresent()) {
                        var camToTag =
                                new Transform3d(
                                        new Pose3d().plus(multiTagResult.get().estimatedPose.best), tagPose.get());
                        // match expected AprilTag coordinate system
                        camToTag =
                                CoordinateSystem.convert(camToTag, CoordinateSystem.NWU(), CoordinateSystem.EDN());
                        // (AprilTag expects Z axis going into tag)
                        camToTag =
                                new Transform3d(
                                        camToTag.getTranslation(),
                                        new Rotation3d(0, Math.PI, 0).plus(camToTag.getRotation()));
                        tagPoseEstimate = new AprilTagPoseEstimate(camToTag, camToTag, 0, 0);
                    }
                }

                // populate the target list
                // Challenge here is that TrackedTarget functions with OpenCV Contour
                TrackedTarget target =
                        new TrackedTarget(
                                detection,
                                tagPoseEstimate,
                                new TargetCalculationParameters(
                                        false, null, null, null, null, frameStaticProperties));

                var correctedBestPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getBestCameraToTarget3d());
                var correctedAltPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getAltCameraToTarget3d());

                target.setBestCameraToTarget3d(
                        new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
                target.setAltCameraToTarget3d(
                        new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

                targetList.add(target);
            }
        }

        // ALGAE
        CVPipeResult<Mat> paddedResult = paddedPipe.run(frame.colorImage.getMat());
        sumPipeNanosElapsed += paddedResult.nanosElapsed;

        CVPipeResult<Mat> hsvResult = hsvPipe.run(paddedResult.output);
        sumPipeNanosElapsed += hsvResult.nanosElapsed;

        CVPipeResult<Mat> maskResult = maskPipe.run(hsvResult.output);
        sumPipeNanosElapsed += maskResult.nanosElapsed;

        CVPipeResult<List<Contour>> contoursResult = findContoursPipe.run(maskResult.output);
        sumPipeNanosElapsed += contoursResult.nanosElapsed;
        List<Contour> contours = contoursResult.output;

        CVPipeResult<Mat> unPadResult = unPadPipe.run(hsvResult.output);

        CVPipeResult<List<AlgaeResult>> algaeDetectionResult = algaeDetectionPipe.run(contours);
        sumPipeNanosElapsed += algaeDetectionResult.nanosElapsed;
        List<AlgaeResult> algaeResults = algaeDetectionResult.output;

        long currentTimeNanos = System.nanoTime();
        List<CVShape> algaeShapes =
                algaeResults.stream().map(AlgaeResult::getShape).collect(Collectors.toList());

        List<PotentialTarget> potentialTargets =
                algaeShapes.stream()
                        .map(
                                shape -> {
                                    return new PotentialTarget(shape.getContour(), shape);
                                })
                        .collect(Collectors.toList());
        sumPipeNanosElapsed += System.nanoTime() - currentTimeNanos;

        CVPipeResult<List<TrackedTarget>> collect2dTargetsResult =
                collect2dTargetsPipe.run(potentialTargets);
        sumPipeNanosElapsed += collect2dTargetsResult.nanosElapsed;

        currentTimeNanos = System.nanoTime();
        List<TrackedTarget> algaeTargets = collect2dTargetsResult.output;

        if (settings.solvePNPEnabled) {
            for (int i = 0; i < algaeTargets.size(); i++) {
                algaeTargets
                        .get(i)
                        .setBestCameraToTarget3d(algaeResults.get(i).getCameraToAlgaeTransform());
                algaeTargets.get(i).setAltCameraToTarget3d(new Transform3d());
            }
        }
        targetList.addAll(algaeTargets);
        unPadResult.output.copyTo(frame.processedImage.getMat());

        sumPipeNanosElapsed += System.nanoTime() - currentTimeNanos;

        // COMMON
        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(
                frame.sequenceID, sumPipeNanosElapsed, fps, targetList, multiTagResult, frame);
    }

    @Override
    public void release() {
        aprilTagDetectionPipe.release();
        singleTagPoseEstimatorPipe.release();
        super.release();
    }
}
