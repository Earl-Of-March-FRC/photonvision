package org.photonvision.vision.pipe.impl;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.photonvision.vision.pipe.CVPipe;

public class PaddedPipe extends CVPipe<Mat, Mat, PaddedPipe.PaddingParams> {

    @Override
    protected Mat process(Mat in) {
        // Apply padding to the image
        Mat paddedImage = new Mat();
        Core.copyMakeBorder(in, paddedImage, params.getPadding(), params.getPadding(), params.getPadding(), params.getPadding(), Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
        return paddedImage;
    }

    // The PaddingParams class contains parameters for the padding
    public static class PaddingParams {
        private final int padding;

        // Constructor to initialize padding size
        public PaddingParams(int padding) {
            this.padding = padding;
        }

        // Getter for padding size
        public int getPadding() {
            return padding;
        }
    }
}
