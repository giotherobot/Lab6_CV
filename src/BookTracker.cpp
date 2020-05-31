#include "BookTracker.h"


void BookTracker::loadTargets(vector<String> target_files)
{
    // Loads targets to track
    targets.resize(target_files.size());
    for (int i = 0; i < target_files.size(); i++)
        targets[i].image = imread(target_files[i]);
}

bool BookTracker::loadVideo(String video_file)
{
    // Loads video
    cap = VideoCapture(video_file);
    if (cap.isOpened())
    {
        cap >> firstFrame.image;
        return true;
    }

    return false;
}

void BookTracker::computeFeaturesOnTargets()
{
    // Computes features on each target
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(nfeaturesTargets);

    for (int i = 0; i < targets.size(); i++)
    {
        sift->detect(targets[i].image, targets[i].keypoints);
        sift->compute(targets[i].image, targets[i].keypoints, targets[i].descriptors);
    }
}

void BookTracker::computeFeaturesOnFrame()
{
    // Computes features on the first video frame
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(nfeaturesFrame);
    sift->detect(firstFrame.image, firstFrame.keypoints);
    sift->compute(firstFrame.image, firstFrame.keypoints, firstFrame.descriptors);
}

homoWithPoints BookTracker::matchTargetToFrame(imageWithFeatures target)
{
    // Mathes the features of the target with the first frame
    BFMatcher matcher(cv::NORM_L2);
    vector<DMatch> matches;
    vector<DMatch> tmp_valid_matches;
    matcher.match(target.descriptors, firstFrame.descriptors, matches);

    //Keep Only matches under a certain threashold
    vector<Point2f> src_kp;
    vector<Point2f> dst_kp;

    //Finding min dist
    float min_dist = matches[0].distance;
    for (int j = 1; j < matches.size(); ++j)
        if (min_dist > matches[j].distance)
            min_dist = matches[j].distance;

    //Refine the matches
    for (int j = 0; j < matches.size(); ++j)
        if (matches[j].distance < min_dist * homoExcludeRatio)
        {
            src_kp.push_back(target.keypoints[j].pt);
            dst_kp.push_back(firstFrame.keypoints[matches[j].trainIdx].pt);
            tmp_valid_matches.push_back(matches[j]);
        }

    return computeHomoAndInliers(src_kp, dst_kp);
}

homoWithPoints BookTracker::computeHomoAndInliers(vector<Point2f> src_kp, vector<Point2f> dst_kp)
{
    // Use of RANSAC to discard outliers and compute homography
    homoWithPoints output;
    Mat in_mask;
    output.homography = findHomography(src_kp, dst_kp, cv::RANSAC, 3, in_mask);

    for (int j = 0; j < in_mask.rows; ++j)
        if (in_mask.at<unsigned short>(j, 0))
            output.points.push_back(dst_kp[j]);

    return output;
}

homoWithPoints BookTracker::computeHomoAndInliers(pointsWithStatus src_kp, pointsWithStatus dst_kp)
{
    vector<Point2f> tmp_src_kp;
    for (int i = 0; i < src_kp.points.size(); i++)
	    if (dst_kp.status[i] == 1)
	    	tmp_src_kp.push_back(src_kp.points[i]);

    return computeHomoAndInliers(tmp_src_kp, dst_kp.points);
}

homoWithPoints BookTracker::excludeOutliers(homoWithPoints homo, vector<Point2f> corners)
{
    // Calculate area of rectangle
    float rectArea = calculateArea(corners);
    vector<Point2f> tmp_points;
    for (int i = 0; i < homo.points.size(); i++)
    {
        // Calculate area of triangles
        float area = 0;
        for (int j = 0; j < corners.size(); j++)
        {
            vector<Point2f> tmp_cor;
            tmp_cor.push_back(corners[j]);
            tmp_cor.push_back(homo.points[i]);
            tmp_cor.push_back(corners[j + 1 < corners.size() ? j + 1 : 0]);
            area += calculateArea(tmp_cor);
        }
        if (area <= rectArea)
            tmp_points.push_back(homo.points[i]);
    }
    homo.points = tmp_points;
    return homo;
}

float BookTracker::calculateArea(vector<Point2f> corners)
{
    float area;
    for (int i = 0; i < corners.size(); i++)
    {
        area += corners[i].x * corners[i + 1 < corners.size() ? i + 1 : 0].y - corners[i].y * corners[i + 1 < corners.size() ? i + 1 : 0].x;
    }
    return abs(area / 2);
}

vector<Point2f> BookTracker::genCornersForTarget(imageWithFeatures target)
{
    // Generates the 4 corners from the target
    vector<Point2f> corners;
    corners.push_back(Point2f(0, 0));
    corners.push_back(Point2f((float)target.image.cols, 0));
    corners.push_back(Point2f((float)target.image.cols, (float)target.image.rows));
    corners.push_back(Point2f(0, (float)target.image.rows));

    return corners;
}

vector<Point2f> BookTracker::updateCorners(vector<Point2f> corners, homoWithPoints homo)
{
    // Computes the corners transform wrt to the homography
    vector<Point2f> new_corners;
    perspectiveTransform(corners, new_corners, homo.homography);

    return new_corners;
}

Mat BookTracker::drawRectangle(Mat src, vector<Point2f> corners, Scalar color)
{
    // Draws a rectangle onthe image src, with corners
    Mat dest = src.clone();
    line(dest, corners[0], corners[1], color, 4);
    line(dest, corners[1], corners[2], color, 4);
    line(dest, corners[2], corners[3], color, 4);
    line(dest, corners[3], corners[0], color, 4);
    return dest;
}

pointsWithStatus BookTracker::computeOptFlow(Mat prevFrame, Mat frame, vector<Point2f> prevPoints)
{
    // Computes the optical flow from prevFrame to frame of the selected points
    Mat grayFrame, grayPrevFrame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cvtColor(prevFrame, grayPrevFrame, cv::COLOR_BGR2GRAY);

    vector<uchar> status;
    vector<float> err;
    TermCriteria term = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

    vector<Point2f> nextKPoints;
    calcOpticalFlowPyrLK(grayPrevFrame, grayFrame, prevPoints, nextKPoints, status, err,
                         Size(15, 15), 3, term, 0);

    pointsWithStatus newKPoints;
    for (size_t i = 0; i < prevPoints.size(); i++)
        if (status[i] == 1)
            newKPoints.points.push_back(nextKPoints[i]);

    newKPoints.status = status;
    return newKPoints;
}

frameWithPointsAndCorners BookTracker::processFrame(Mat frame, frameWithPointsAndCorners prevFrame, Scalar color)
{
    // Processes the frame, ie computes the optical flow, the homography and draws the rectangle.
    frameWithPointsAndCorners newFrame;
    newFrame.points = computeOptFlow(prevFrame.frame, frame, prevFrame.points.points);

    homoWithPoints newHomo = computeHomoAndInliers(prevFrame.points, newFrame.points);
    newHomo = excludeOutliers(newHomo, prevFrame.corners);
    newFrame.corners = updateCorners(prevFrame.corners, newHomo);
    newFrame.frame = frame.clone();

    Mat img = drawRectangle(frame, newFrame.corners, color);
    return newFrame;
}

void BookTracker::setup()
{
    // Initial operations on the targets and first frame.
    computeFeaturesOnTargets();
    computeFeaturesOnFrame();

    for (int i = 0; i < targets.size(); i++)
    {
        homo.push_back(matchTargetToFrame(targets[i]));

        vector<Point2f> corners = genCornersForTarget(targets[i]);
        corners = updateCorners(corners, homo[i]);
        homo[i] = excludeOutliers(homo[i], corners);

        Mat img = drawRectangle(firstFrame.image, corners, colors[i%colors.size()]);
        drawTrackedFeatures(img, homo[i].points, colors[i%colors.size()]);

        // Shows the rectangles computed
        
        namedWindow("Targets", WINDOW_NORMAL);
        resizeWindow("Targets", 600, 600);
        imshow("Targets", img);
        waitKey(0);

        this->corners.push_back(corners);
    }

}

void BookTracker::loop()
{
    // Loops the video and computes each frame.
    Mat frame = firstFrame.image.clone();
    vector<frameWithPointsAndCorners> prevFrames;
    for (int i = 0; i < targets.size(); i++)
    {
        frameWithPointsAndCorners tmpFrame;
        tmpFrame.frame = firstFrame.image.clone();
        tmpFrame.points.points = homo[i].points;
        tmpFrame.corners = corners[i];
        prevFrames.push_back(tmpFrame);
    }

    Mat displayFrame;
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        displayFrame = frame.clone();
        for (int i = 0; i < targets.size(); i++)
        {
            prevFrames[i] = processFrame(frame, prevFrames[i], colors[i%colors.size()]);
            displayFrame = drawRectangle(displayFrame, prevFrames[i].corners, colors[i%colors.size()]);
            drawTrackedFeatures(displayFrame, prevFrames[i].points.points, colors[i%colors.size()]);

        }
        imshow("Video", displayFrame);
        waitKey(30);
    }
}


void BookTracker::drawTrackedFeatures(Mat img, vector<Point2f> features, Scalar color)
{
	//Points that we are tracking
	for ( int i = 0 ; i < features.size(); ++i )
		circle(img, features[i], 3, color);
}


