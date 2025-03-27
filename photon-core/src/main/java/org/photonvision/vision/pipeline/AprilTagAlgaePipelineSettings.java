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

import com.fasterxml.jackson.annotation.JsonTypeName;
import java.util.Objects;
import org.photonvision.common.util.numbers.IntegerCouple;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.target.TargetModel;

@JsonTypeName("AprilTagAlgaePipelineSettings")
public class AprilTagAlgaePipelineSettings extends AdvancedPipelineSettings {
    // APRIL TAG
    public AprilTagFamily tagFamily = AprilTagFamily.kTag36h11;
    public int decimate = 1;
    public double blur = 0;
    public int threads = 4; // Multiple threads seems to be better performance on most platforms
    public boolean debug = false;
    public boolean refineEdges = true;
    public int numIterations = 40;
    public int hammingDist = 0;
    public int decisionMargin = 35;
    public boolean doMultiTarget = false;
    public boolean doSingleTargetAlways = false;

    // ALGAE
    public IntegerCouple circularity = new IntegerCouple(30, 100);
    public int padding = 20;
    public int erosion = 2;
    public int initialDilation = 2;
    public IntegerCouple edgeThresholds = new IntegerCouple(100, 300);
    public int edgeDilation = 3;
    public int finalDilation = 3;

    public AprilTagAlgaePipelineSettings() {
        super();
        pipelineType = PipelineType.AprilTagAlgae;

        // APRIL TAG
        outputShowMultipleTargets = true;
        targetModel = TargetModel.kAprilTag6p5in_36h11;
        cameraAutoExposure = false;
        ledMode = false;

        // COMMON
        cameraExposureRaw = 20;
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                super.hashCode(),
                tagFamily,
                decimate,
                blur,
                threads,
                debug,
                refineEdges,
                numIterations,
                hammingDist,
                decisionMargin,
                doMultiTarget,
                doSingleTargetAlways,
                circularity,
                padding,
                erosion,
                initialDilation,
                edgeThresholds,
                edgeDilation,
                finalDilation);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!super.equals(obj)) return false;
        if (getClass() != obj.getClass()) return false;
        AprilTagAlgaePipelineSettings other = (AprilTagAlgaePipelineSettings) obj;
        return Objects.equals(tagFamily, other.tagFamily)
                && decimate == other.decimate
                && Double.doubleToLongBits(blur) == Double.doubleToLongBits(other.blur)
                && threads == other.threads
                && debug == other.debug
                && refineEdges == other.refineEdges
                && numIterations == other.numIterations
                && hammingDist == other.hammingDist
                && decisionMargin == other.decisionMargin
                && doMultiTarget == other.doMultiTarget
                && doSingleTargetAlways == other.doSingleTargetAlways
                && Objects.equals(circularity, other.circularity)
                && padding == other.padding
                && erosion == other.erosion
                && initialDilation == other.initialDilation
                && Objects.equals(edgeThresholds, other.edgeThresholds)
                && edgeDilation == other.edgeDilation
                && finalDilation == other.finalDilation;
    }
}
