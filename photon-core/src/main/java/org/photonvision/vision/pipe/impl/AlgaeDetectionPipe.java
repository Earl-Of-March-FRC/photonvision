// TODO: Make a blur pipe, pipe the org image to the hsv pipe and pip that to algae. Keep in mind unit conversions

package org.photonvision.vision.pipe.impl;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.RotatedRect;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.Contour;
import org.photonvision.vision.pipe.CVPipe;

import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;

public class AlgaeDetectionPipe extends CVPipe<List<Contour>, List<AlgaeDetectionPipe.AlgaeResult>, AlgaeDetectionPipe.AlgaeDetectionParams> {
    // // Constants
    // private static final double KNOWN_DIAMETER = 475.00; // mm

    // // Contour Conditionals
    // private static final int MIN_AREA = 3000;
    // private static final double MIN_CIRCULARITY = 0.3;

    private static final Mat CAMERA_MATRIX = new Mat(3, 3, CvType.CV_64F);

    private ObjectDetection detector;
    private ScreenToWorldConverter converter;

    @Override
    public void setParams(AlgaeDetectionParams params) {
        super.setParams(params);
    
        // Algae Pipes
        detector = new ObjectDetection(params.getMinArea(), params.getMinCircularity());
        
        if (params.getCameraCalibration() != null) {
            for (int i = 0; i < CAMERA_MATRIX.rows(); i++) {
                for (int j = 0; j < CAMERA_MATRIX.cols(); j++) {
                    CAMERA_MATRIX.put(i, j, params.getCameraCalibration().getIntrinsicsArr()[i * CAMERA_MATRIX.cols() + j]);
                }
            }
        }

        converter = new ScreenToWorldConverter(params.getDiameter(), CAMERA_MATRIX);
    }

    public static class AlgaeDetectionParams {
        private final double diameter_mm;
        private final int min_area;
        private final double min_circularity;
        private final CameraCalibrationCoefficients cameraCalibration;

        public AlgaeDetectionParams(double diameter_mm, int min_area, double min_circularity, CameraCalibrationCoefficients cameraCalibration) {
            this.diameter_mm = diameter_mm;
            this.min_area = min_area;
            this.min_circularity = min_circularity;
            this.cameraCalibration = cameraCalibration;
        }

        public double getDiameter() {
            return diameter_mm;
        }

        public int getMinArea() {
            return min_area;
        }

        public double getMinCircularity() {
            return min_circularity;
        }

        public CameraCalibrationCoefficients getCameraCalibration() {
            return cameraCalibration;
        }
    }

    @Override
    protected List<AlgaeResult> process(List<Contour> in) {
        Optional<AlgaeResult> result = detector.findLargestAlgae(in);
        if (result.isPresent()) {
            Point algaeCenter = result.get().getCenter();
            double algaeRadius = result.get().getRadius();

            return List.of(new AlgaeResult(algaeCenter, algaeRadius));
        }
        return List.of();
    }

    public class AlgaeResult {
        private Point center;
        private double radius;
    
        public AlgaeResult(Point center, double radius) {
            this.center = center;
            this.radius = radius;
        }
    

        public Point getCenter() {
            return center;
        }
    
        public double getRadius() {
            return radius;
        }

        public double getDistance() {
            return (converter.calculateDistance(radius * 2)) / 10; // in m
        }

        public double getXAngle() {
            return converter.calculateHorizontalAngle(center.x);
        }

        public double getYAngle() {
            return converter.calculateVerticalAngle(center.y);
        }

        public Contour geContour() {
            return new Contour(new Rect2d(center.x-radius, center.y-radius, radius*2, radius*2));
        }

        public CVShape getShape() {
            return new CVShape(geContour(), center, radius);
        }

        public Transform3d getCameraToAlgaeTransform() {
            double xTranslation = getDistance() * Math.cos(Math.toRadians(getYAngle())) * Math.cos(Math.toRadians(getXAngle()));
            double yTranslation = -1 * getDistance() * Math.cos(Math.toRadians(getYAngle())) * Math.sin(Math.toRadians(getXAngle()));
            double zTranslation = getDistance() * Math.sin(Math.toRadians(getYAngle()));
            return new Transform3d(xTranslation, yTranslation, zTranslation, new Rotation3d());
        }
    }

    public class ObjectDetection {
        private int minArea;
        private double minCircularity;

        public ObjectDetection( int minArea, double minCircularity) {
            this.minArea = minArea;
            this.minCircularity = minCircularity;
        }

        public Optional<AlgaeResult> findLargestAlgae(List<Contour> contoursList) {

            List<MatOfPoint> contours = contoursList.stream()
            .map(contour -> contour.mat)  // Extract the MatOfPoint from each Contour
            .collect(Collectors.toList());  // Collect into a List<MatOfPoint>

            // Variables to store the largest algae
            Point largestBallCenter = null;
            double largestArea = 0;
            double largestRadius = 0;

            // Iterate over the contours
            for (MatOfPoint contour : contours) {

                double area = Imgproc.contourArea(contour);

                // Convert MatOfPoint to MatOfPoint2f for arcLength
                MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());

                double perimeter = Imgproc.arcLength(contour2f, true);
                double circularity = (perimeter > 0) ? (4 * Math.PI * area / (perimeter * perimeter)) : 0;

                if (area > minArea && circularity > minCircularity) {
                    // Approximate the contour to a circle
                    RotatedRect minEnclosingCircle = Imgproc.minAreaRect(new MatOfPoint2f(contour.toArray()));
                    Point center = minEnclosingCircle.center;
                    double radius = minEnclosingCircle.size.height / 2;

                    if (radius > 10) {
                        // Check if this is the largest algae ball so far
                        if (area > largestArea) {
                            largestArea = area;
                            largestBallCenter = center;
                            largestRadius = radius;
                        }
                    }
                }

            }

            if (largestBallCenter == null) {
                return Optional.empty();
            }
            return Optional.of(new AlgaeResult(largestBallCenter, largestRadius));

        }
    }

    static class ScreenToWorldConverter {
        double objectRealWidth;
        Mat CAMERA_MATRIX;

        public ScreenToWorldConverter(double objectRealWidth, Mat cameraMatrix) {
            this.objectRealWidth = objectRealWidth;
            this.CAMERA_MATRIX = cameraMatrix; // Initialize with passed camera matrix
        }

        public double calculateDistance(double detectionWidth) {
            double fx = CAMERA_MATRIX.get(0, 0)[0];
            return (((objectRealWidth * fx) / detectionWidth) - 20) / 100; // m
        }

        public double calculateHorizontalAngle(double objectCenterX) {
            // Extract focal length (fx) and principal point (cx) from the camera matrix
            double fx = CAMERA_MATRIX.get(0, 0)[0];
            double cx = CAMERA_MATRIX.get(0, 2)[0];
        
            // Compute the normalized x coordinate: (u - cx) / fx
            double normalizedX = (objectCenterX - cx) / fx;
        
            // Calculate the horizontal angle in radians and convert to degrees
            double angleRad = Math.atan(normalizedX);
            double angleDeg = Math.toDegrees(angleRad);
        
            return angleDeg;
        }

        public double calculateVerticalAngle(double objectCenterY) {
            // Extract focal length (fy) and principal point (cy) from the camera matrix
            double fy = CAMERA_MATRIX.get(1, 1)[0];
            double cy = CAMERA_MATRIX.get(1, 2)[0];
        
            // Compute the normalized y coordinate: (objectCenterY - cy) / fy
            double normalizedY = (objectCenterY - cy) / fy;
        
            // Calculate the vertical angle in radians and convert to degrees
            double angleRad = Math.atan(normalizedY);
            double realAngle = Math.toDegrees(angleRad);
        
            return realAngle;
        }

        // public double calculateHorizontalAngle(Mat frame, double objectCenterX, double cameraOffset) {
        //     double screenCenterX = frame.width() / 2;
        //     double screenCenterY = frame.height() / 2;

        //     // Adjust the object center x-coordinate based on camera offset
        //     // objectCenterX -= cameraOffset; // offset in mm

        //     Mat matInverted = new Mat();
        //     Core.invert(CAMERA_MATRIX, matInverted);

        //     // Calculate vector1 and vector2
        //     MatOfFloat vector1 = new MatOfFloat((float) objectCenterX, (float) screenCenterY, 1.0f);
        //     MatOfFloat vector2 = new MatOfFloat((float) screenCenterX, (float) screenCenterY, 1.0f);

        //     // Convert MatOfFloat to float array
        //     float[] vec1Arr = vector1.toArray();
        //     float[] vec2Arr = vector2.toArray();

        //     // Perform the dot product and angle calculation
        //     double dotProduct = vec1Arr[0] * vec2Arr[0] + vec1Arr[1] * vec2Arr[1] + vec1Arr[2] * vec2Arr[2];

        //     double norm1 = Math.sqrt(vec1Arr[0] * vec1Arr[0] + vec1Arr[1] * vec1Arr[1] + vec1Arr[2] * vec1Arr[2]);
        //     double norm2 = Math.sqrt(vec2Arr[0] * vec2Arr[0] + vec2Arr[1] * vec2Arr[1] + vec2Arr[2] * vec2Arr[2]);

        //     double cosAngle = dotProduct / (norm1 * norm2);
        //     double realAngle = Math.toDegrees(Math.acos(cosAngle));

        //     if (objectCenterX < screenCenterX) {
        //         realAngle *= -1;
        //     }

        //     return realAngle;
        // }

        // public double calculateVerticalAngle(Mat frame, double objectCenterY, double cameraOffset) {
        //     double screenCenterX = frame.width() / 2;
        //     double screenCenterY = frame.height() / 2;

        //     // Adjust the object center y-coordinate based on camera offset
        //     // objectCenterY -= cameraOffset; // offset in mm\

        //     // Calculate vector1 and vector2
        //     MatOfFloat vector1 = new MatOfFloat((float) screenCenterX, (float) objectCenterY, 1.0f);
        //     MatOfFloat vector2 = new MatOfFloat((float) screenCenterX, (float) screenCenterY, 1.0f);

        //     // Convert MatOfFloat to float array
        //     float[] vec1Arr = vector1.toArray();
        //     float[] vec2Arr = vector2.toArray();

        //     // Perform the dot product and angle calculation
        //     double dotProduct = vec1Arr[0] * vec2Arr[0] + vec1Arr[1] * vec2Arr[1] + vec1Arr[2] * vec2Arr[2];

        //     double norm1 = Math.sqrt(vec1Arr[0] * vec1Arr[0] + vec1Arr[1] * vec1Arr[1] + vec1Arr[2] * vec1Arr[2]);
        //     double norm2 = Math.sqrt(vec2Arr[0] * vec2Arr[0] + vec2Arr[1] * vec2Arr[1] + vec2Arr[2] * vec2Arr[2]);

        //     double cosAngle = dotProduct / (norm1 * norm2);
        //     double realAngle = Math.toDegrees(Math.acos(cosAngle));

        //     if (objectCenterY < screenCenterY) {
        //         realAngle *= -1;
        //     }

        //     return -realAngle;
        // }
    }
}