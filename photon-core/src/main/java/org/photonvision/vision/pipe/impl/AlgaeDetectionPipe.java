// TODO: Make a blur pipe, pipe the org image to the hsv pipe and pip that to algae. Keep in mind unit conversions 
package org.photonvision.vision.pipe.impl;

import java.util.List;
import java.util.Optional;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect2d;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.vision.opencv.CVShape;
import org.photonvision.vision.opencv.Contour;
import org.photonvision.vision.pipe.CVPipe;

import edu.wpi.first.math.geometry.Rotation3d;
import edu.wpi.first.math.geometry.Transform3d;

public class AlgaeDetectionPipe extends CVPipe<Mat, List<AlgaeDetectionPipe.AlgaePose>, AlgaeDetectionPipe.AlgaeDetectionParams> {

    // Constants
    private static final double KNOWN_DIAMETER = 475.00; // mm

    private static final Scalar LOWER_BALL = new Scalar(80, 50, 60); // HSV lower-bound
    private static final Scalar UPPER_BALL = new Scalar(100, 255, 255); // HSV upper-bound

    // Contour Conditionals
    private static final int MIN_AREA = 3000;
    private static final double MIN_CIRCULARITY = 0.3;

    private static final int PADDING = 50;

    private static final Mat CAMERA_MATRIX = new Mat(3, 3, CvType.CV_64F);

    private final ObjectDetection detector;
    private final Computation computation; 

    public AlgaeDetectionPipe() {
        super();
        CAMERA_MATRIX.put(0, 0, 1413.70008, 0, 672.90);
        CAMERA_MATRIX.put(1, 0, 0, 1437.31, 415.13);
        CAMERA_MATRIX.put(2, 0, 0, 0, 1);

        detector = new ObjectDetection(LOWER_BALL, UPPER_BALL, MIN_AREA, MIN_CIRCULARITY);
        computation = new Computation(CAMERA_MATRIX.get(0,0)[0], KNOWN_DIAMETER, CAMERA_MATRIX);
    }

    public class AlgaePose {
        private Point algaeCenter;
        private double algaeRadius;
        private Mat mat;

        public AlgaePose(Point algaeCenter, double algaeRadius, Mat in) {
            this.algaeCenter = algaeCenter;
            this.algaeRadius = algaeRadius;
            this.mat = in;
        }

        public double getDistance() {
            return (computation.calculateDistance(algaeRadius * 2)) / 10; // in cm
        }

        public double getXAngle() {
            return computation.calculateHorizontalAngle(mat, algaeCenter.x, 0);
        }

        public double getYAngle() {
            return computation.calculateVerticalAngle(mat, algaeCenter.y, 0);
        }

        public Contour geContour() {
            return new Contour(new Rect2d(algaeCenter.x-algaeRadius, algaeCenter.y-algaeRadius, algaeRadius*2, algaeRadius*2));
        }

        public CVShape getShape() {
            return new CVShape(geContour(), algaeCenter, algaeRadius);
        }

        public Transform3d getCameraToAlgaeTransform() {
            double xTranslation = getDistance() * Math.cos(Math.toRadians(getYAngle())) * Math.cos(Math.toRadians(getXAngle()));
            double yTranslation = -1 * getDistance() * Math.cos(Math.toRadians(getYAngle())) * Math.sin(Math.toRadians(getXAngle()));
            double zTranslation = getDistance() * Math.sin(Math.toRadians(getYAngle()));
            return new Transform3d(xTranslation, yTranslation, zTranslation, new Rotation3d());
        }
    }

    @Override
    protected List<AlgaePose> process(Mat in) {
        Mat outputMat = in.clone();
        Optional<AlgaeResult> result = detector.findLargestAlgae(outputMat);
        if (result.isPresent()) {
            Point algaeCenter = result.get().getCenter();
            double algaeRadius = result.get().getRadius();

            // // Calculate distance and angles
            // double distance = (computation.calculateDistance(algaeRadius * 2)) / 10; // in cm
            // double x_angle = computation.calculateHorizontalAngle(in, algaeCenter.x, 45.7);
            // double y_angle = computation.calculateVerticalAngle(in, algaeCenter.y, 65);

            // Optionally, display some information on the image
            // Imgproc.putText(outputMat, "Distance: " + distance + " cm", new Point(50, 50),
            //         Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 0), 2);
            // Imgproc.putText(outputMat, "X Angle: " + x_angle + " degrees", new Point(50, 100),
            //         Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 0), 2);
            // Imgproc.putText(outputMat, "Y Angle: " + y_angle + " degrees", new Point(50, 150),
            //         Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 0), 2);

            int unpaddedX = (int) algaeCenter.x - PADDING;
            int unpaddedY = (int) algaeCenter.y - PADDING;
            // Imgproc.circle(outputMat, new Point(unpaddedX, unpaddedY), (int) algaeRadius, new Scalar(255, 0, 255), 5);
            return List.of(new AlgaePose(new Point(unpaddedX, unpaddedY), algaeRadius, in));
        }
        return List.of();
    }

    public static class AlgaeDetectionParams {
    }

    public static class AlgaeResult {
        private Mat image;
        private Point center;
        private double radius;
    
        public AlgaeResult(Mat image, Point center, double radius) {
            this.image = image;
            this.center = center;
            this.radius = radius;
        }
    
        public Mat getImage() {
            return image;
        }
    
        public Point getCenter() {
            return center;
        }
    
        public double getRadius() {
            return radius;
        }
    }

    public static class ObjectDetection {
        private Scalar lowerBound;
        private Scalar upperBound;
        private int minArea;
        private double minCircularity;

        public ObjectDetection(Scalar lowerBound, Scalar upperBound, int minArea, double minCircularity) {
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
            this.minArea = minArea;
            this.minCircularity = minCircularity;
        }

        public Optional<AlgaeResult> findLargestAlgae(Mat image) {

            // Create a padded version of the image to handle partial objects near the
            // borders
            Mat paddedImage = new Mat();
            Core.copyMakeBorder(image, paddedImage, PADDING, PADDING, PADDING, PADDING, Core.BORDER_CONSTANT,
                    new Scalar(0, 0, 0));

            // Convert to HSV
            Mat hsv = new Mat();
            Imgproc.cvtColor(paddedImage, hsv, Imgproc.COLOR_BGR2HSV);

            // Apply Gaussian Blur
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(hsv, blurred, new Size(9, 9), 0);

            // Create a mask for the algae color
            Mat mask = new Mat();
            Core.inRange(blurred, lowerBound, upperBound, mask);

            // Morphological operations
            Imgproc.erode(mask, mask, new Mat(), new Point(-1, -1), 2);
            Imgproc.dilate(mask, mask, new Mat(), new Point(-1, -1), 2);

            // Canny edge detection
            Mat edges = new Mat();
            Imgproc.Canny(mask, edges, 100, 300);

            // Dilate the edges to connect broken parts
            Mat dilatedEdges = new Mat();
            Imgproc.dilate(edges, dilatedEdges, new Mat(), new Point(-1, -1), 3);

            // Further dilation to fill the edges
            Mat filledEdges = new Mat();
            Imgproc.dilate(dilatedEdges, filledEdges, new Mat(), new Point(-1, -1), 3);

            // Find contours
            java.util.List<MatOfPoint> contours = new java.util.ArrayList<>();
            Imgproc.findContours(filledEdges, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

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
            return Optional.of(new AlgaeResult(image, largestBallCenter, largestRadius));

        }
    }

    static class Computation {
        double focalLengthX;
        double objectRealWidth;
        static Mat CAMERA_MATRIX;

        @SuppressWarnings("static-access")
        public Computation(double focalLengthX, double objectRealWidth, Mat cameraMatrix) {
            this.focalLengthX = focalLengthX;
            this.objectRealWidth = objectRealWidth;
            this.CAMERA_MATRIX = cameraMatrix; // Initialize with passed camera matrix
        }

        public double calculateDistance(double detectionWidth) {
            return (((objectRealWidth * focalLengthX) / detectionWidth) - 20) / 100; // m
        }

        public double calculateHorizontalAngle(Mat frame, double objectCenterX, double cameraOffset) {
            try {
                double screenCenterX = frame.width() / 2;
                double screenCenterY = frame.height() / 2;

                // Adjust the object center x-coordinate based on camera offset
                // objectCenterX -= cameraOffset; // offset in mm

                Mat matInverted = new Mat();
                Core.invert(CAMERA_MATRIX, matInverted);

                // Calculate vector1 and vector2
                MatOfFloat vector1 = new MatOfFloat((float) objectCenterX, (float) screenCenterY, 1.0f);
                MatOfFloat vector2 = new MatOfFloat((float) screenCenterX, (float) screenCenterY, 1.0f);

                // Convert MatOfFloat to float array
                float[] vec1Arr = vector1.toArray();
                float[] vec2Arr = vector2.toArray();

                // Perform the dot product and angle calculation
                double dotProduct = vec1Arr[0] * vec2Arr[0] + vec1Arr[1] * vec2Arr[1] + vec1Arr[2] * vec2Arr[2];

                double norm1 = Math.sqrt(vec1Arr[0] * vec1Arr[0] + vec1Arr[1] * vec1Arr[1] + vec1Arr[2] * vec1Arr[2]);
                double norm2 = Math.sqrt(vec2Arr[0] * vec2Arr[0] + vec2Arr[1] * vec2Arr[1] + vec2Arr[2] * vec2Arr[2]);

                double cosAngle = dotProduct / (norm1 * norm2);
                double realAngle = Math.toDegrees(Math.acos(cosAngle));

                if (objectCenterX < screenCenterX) {
                    realAngle *= -1;
                }

                return realAngle;

            } catch (Exception e) {
                System.out.println("Error occurred while calculating horizontal angle");
                return 0.0;
            }
        }

        public double calculateVerticalAngle(Mat frame, double objectCenterY, double cameraOffset) {
            try {
                double screenCenterX = frame.width() / 2;
                double screenCenterY = frame.height() / 2;

                // Adjust the object center y-coordinate based on camera offset
                // objectCenterY -= cameraOffset; // offset in mm

                Mat matInverted = new Mat();
                Core.invert(CAMERA_MATRIX, matInverted); // Invert camera matrix

                // Calculate vector1 and vector2
                MatOfFloat vector1 = new MatOfFloat((float) screenCenterX, (float) objectCenterY, 1.0f);
                MatOfFloat vector2 = new MatOfFloat((float) screenCenterX, (float) screenCenterY, 1.0f);

                // Convert MatOfFloat to float array
                float[] vec1Arr = vector1.toArray();
                float[] vec2Arr = vector2.toArray();

                // Perform the dot product and angle calculation
                double dotProduct = vec1Arr[0] * vec2Arr[0] + vec1Arr[1] * vec2Arr[1] + vec1Arr[2] * vec2Arr[2];

                double norm1 = Math.sqrt(vec1Arr[0] * vec1Arr[0] + vec1Arr[1] * vec1Arr[1] + vec1Arr[2] * vec1Arr[2]);
                double norm2 = Math.sqrt(vec2Arr[0] * vec2Arr[0] + vec2Arr[1] * vec2Arr[1] + vec2Arr[2] * vec2Arr[2]);

                double cosAngle = dotProduct / (norm1 * norm2);
                double realAngle = Math.toDegrees(Math.acos(cosAngle));

                if (objectCenterY < screenCenterY) {
                    realAngle *= -1;
                }

                return -realAngle;

            } catch (Exception e) {
                System.out.println("Error occurred while calculating vertical angle");
                return 0.0;
            }
        }
    }
}
