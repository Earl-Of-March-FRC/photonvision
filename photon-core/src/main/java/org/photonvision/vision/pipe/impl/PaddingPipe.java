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

package org.photonvision.vision.pipe.impl;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.photonvision.vision.pipe.CVPipe;

public class PaddingPipe extends CVPipe<Mat, Mat, PaddingPipe.PaddingParams> {
    @Override
    protected Mat process(Mat in) {
        // Apply padding to the image
        Mat paddedImage = new Mat();
        Core.copyMakeBorder(
                in,
                paddedImage,
                params.getPadding(),
                params.getPadding(),
                params.getPadding(),
                params.getPadding(),
                Core.BORDER_CONSTANT,
                new Scalar(0, 0, 0));
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
