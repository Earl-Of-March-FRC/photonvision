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

@JsonTypeName("AlgaePipelineSettings")
public class AlgaePipelineSettings extends AdvancedPipelineSettings {
    public IntegerCouple circularity = new IntegerCouple(30, 100);
    public int padding = 20;
    public int erosion = 2;
    public int initialDilation = 2;
    public IntegerCouple edgeThresholds = new IntegerCouple(100, 300);
    public int edgeDilation = 3;
    public int finalDilation = 3;

    public AlgaePipelineSettings() {
        super();
        pipelineType = PipelineType.Algae;
        cameraExposureRaw = 20;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;
        AlgaePipelineSettings that = (AlgaePipelineSettings) o;
        return Objects.equals(circularity, that.circularity)
                && padding == that.padding
                && erosion == that.erosion
                && initialDilation == that.initialDilation
                && Objects.equals(edgeThresholds, that.edgeThresholds)
                && edgeDilation == that.edgeDilation
                && finalDilation == that.finalDilation;
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                super.hashCode(),
                circularity,
                padding,
                erosion,
                initialDilation,
                edgeThresholds,
                edgeDilation,
                finalDilation);
    }
}
