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

import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.photonvision.vision.pipe.CVPipe;

public class UnPadPipe extends CVPipe<Mat, Mat, UnPadPipe.UnPadParams> {
    @Override
    protected Mat process(Mat in) {
        // Remove padding from the image
        int padding = params.getPadding();
        int newWidth = in.width() - 2 * padding;
        int newHeight = in.height() - 2 * padding;

        // Ensure the new dimensions are valid
        if (newWidth <= 0 || newHeight <= 0) {
            throw new IllegalArgumentException("Padding is too large for the given image dimensions.");
        }

        // Define the region of interest (ROI) to crop the image
        Rect roi = new Rect(padding, padding, newWidth, newHeight);
        Mat croppedImage = new Mat(in, roi);

        return croppedImage;
    }

    // The UnPadParams class contains parameters for the un=padding
    public static class UnPadParams {
        private final int padding;

        // Constructor to initialize padding size
        public UnPadParams(int padding) {
            this.padding = padding;
        }

        // Getter for padding size
        public int getPadding() {
            return padding;
        }
    }
}
