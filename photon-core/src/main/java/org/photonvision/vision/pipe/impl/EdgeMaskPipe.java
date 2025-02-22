package org.photonvision.vision.pipe.impl;

import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.pipe.CVPipe;

public class EdgeMaskPipe extends CVPipe<Mat, Mat, EdgeMaskPipe.MaskParams> {

    @Override
    protected Mat process(Mat inputMask) {
        // Apply morphological transformations to clean up the mask
        applyMorphologicalTransformations(inputMask);

        // Perform Canny edge detection
        Mat edges = detectEdges(inputMask);

        // Dilate edges to connect broken parts and fill gaps
        Mat filledEdges = dilateEdges(edges);

        return filledEdges;
    }

    private void applyMorphologicalTransformations(Mat mask) {
        // Erosion to remove noise
        Imgproc.erode(mask, mask, new Mat(), new Point(-1, -1), params.getErosionIterations());

        // Initial dilation to expand the regions of interest
        Imgproc.dilate(mask, mask, new Mat(), new Point(-1, -1), params.getInitialDilationIterations());
    }

    private Mat detectEdges(Mat mask) {
        // Perform Canny edge detection
        Mat edges = new Mat();
        Imgproc.Canny(mask, edges, params.getEdgeThresholdLow(), params.getEdgeThresholdHigh());
        return edges;
    }

    private Mat dilateEdges(Mat edges) {
        // Dilate the edges to fill gaps and connect broken parts
        Mat dilatedEdges = new Mat();
        Imgproc.dilate(edges, dilatedEdges, new Mat(), new Point(-1, -1), params.getEdgeDilationIterations());

        // Further dilation to ensure continuity of the edges
        Mat filledEdges = new Mat();
        Imgproc.dilate(dilatedEdges, filledEdges, new Mat(), new Point(-1, -1), params.getFinalDilationIterations());

        return filledEdges;
    }

    // Parameters for masking algae
    public static class MaskParams {
        private final int erosionIterations;
        private final int edgeThresholdLow;
        private final int edgeThresholdHigh;
        private final int initialDilationIterations;
        private final int edgeDilationIterations;
        private final int finalDilationIterations;

        // Constructor for setting the iterations
        public MaskParams(int erosion, int initialDilate, int edgeLow, int edgeHigh, int edgeDilate, int finalDilate) {
            this.erosionIterations = erosion;
            this.initialDilationIterations = initialDilate;
            this.edgeThresholdLow = edgeLow;
            this.edgeThresholdHigh = edgeHigh;
            this.edgeDilationIterations = edgeDilate;
            this.finalDilationIterations = finalDilate;
        }

        public int getErosionIterations() {
            return this.erosionIterations;
        }

        public int getInitialDilationIterations() {
            return this.initialDilationIterations;
        }

        public int getEdgeDilationIterations() {
            return this.edgeDilationIterations;
        }

        public int getFinalDilationIterations() {
            return this.finalDilationIterations;
        }

        public int getEdgeThresholdLow() {
            return this.edgeThresholdLow;
        }

        public int getEdgeThresholdHigh() {
            return this.edgeThresholdHigh;
        }
    }
}