package org.photonvision.vision.pipeline;

import com.fasterxml.jackson.annotation.JsonTypeName;
import java.util.Objects;

@JsonTypeName("AlgaePipelineSettings")
public class AlgaePipelineSettings extends AdvancedPipelineSettings{
    
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
        // AlgaePipelineSettings that = (AlgaePipelineSettings) o;
        return true;
    }

    @Override
    public int hashCode() {
        return Objects.hash(super.hashCode());
    }
}
