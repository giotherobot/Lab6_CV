#include <BookTracker.h>

using namespace cv;
using namespace std;

void BookTracker::loadTargets(vector<String> target_files)
{
    targets.resize(target_files.size());
    for (int i = 0; i < target_files.size(); i++)
        targets[i].image = imread(target_files[i]);
}

void BookTracker::loadVideo(String video_file)
{
    cap = VideoCapture(video_file);
    if (cap.isOpened())
        cap >> firstFrame.image;
}


void BookTracker::computeFeaturesOnTargets()
{
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(nfeaturesTargets);

    for (int i = 0; i < targets.size(); i++)
    {
        sift->detect(targets[i].image, targets[i].keypoints);
        sift->compute(targets[i].image, targets[i].keypoints, targets[i].descriptors);
    }
}

void BookTracker::computeFeaturesOnFrame()
{
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(nfeaturesFrame);
    sift->detect(firstFrame.image, firstFrame.keypoints);
    sift->compute(firstFrame.image, firstFrame.keypoints, firstFrame.descriptors);
}

homoWithPoints BookTracker::matchTargetToFrame(imageWithFeatures target)
{
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
    // Use of RANSAC to discard outliers
    homoWithPoints output;
    Mat in_mask;
    output.homography = findHomography(src_kp, dst_kp, cv::RANSAC, 3, in_mask);

    for (int j = 0; j < in_mask.rows; ++j)
        if (in_mask.at<unsigned short>(j, 0))
            output.points.push_back(dst_kp[j]);

    return output;
}

vector<Point2f> BookTracker::genCornersForTarget(imageWithFeatures target)
{
    vector<Point2f> corners;
    corners.push_back(Point2f(0, 0));
    corners.push_back(Point2f((float)target.image.cols, 0));
    corners.push_back(Point2f((float)target.image.cols, (float)target.image.rows));
    corners.push_back(Point2f(0, (float)target.image.rows));

    return corners;
}

vector<Point2f> BookTracker::updateCorners(vector<Point2f> corners, homoWithPoints homo)
{
    vector<Point2f> new_corners;
    perspectiveTransform(corners, new_corners, homo.homography);

    return new_corners;
}

Mat BookTracker::drawRectangle(Mat src, vector<Point2f> corners)
{
    Mat dest = src.clone();
    line(dest, corners[0], corners[1], Scalar(0, 255, 0), 4);
    line(dest, corners[1], corners[2], Scalar(0, 255, 0), 4);
    line(dest, corners[2], corners[3], Scalar(0, 255, 0), 4);
    line(dest, corners[3], corners[0], Scalar(0, 255, 0), 4);
    return dest;
}

vector<Point2f> BookTracker::computeOptFlow(Mat prevFrame, Mat frame, vector<Point2f> prevKPoints)
{   
    Mat grayFrame, grayPrevFrame;
    cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    cvtColor(prevFrame, grayPrevFrame, cv::COLOR_BGR2GRAY);

    vector<uchar> status;
    vector<float> err;
    TermCriteria term = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
    
    vector<Point2f> nextKPoints;
    calcOpticalFlowPyrLK(grayPrevFrame, grayFrame, prevKPoints, nextKPoints, status, err,
    				 Size(21, 21), 3, term, 0);

    vector<Point2f> newKPoints;
    for (size_t i = 0; i < prevKPoints.size(); i++)
    	if (status[i] == 1)
    		newKPoints.push_back(nextKPoints[i]);

    return newKPoints;
}

frameWithPointsAndCorners BookTracker::processFrame(Mat frame, frameWithPointsAndCorners prevFrame)
{   
    frameWithPointsAndCorners newFrame;
    newFrame.points = computeOptFlow(prevFrame.frame, frame, prevFrame.points);

    homoWithPoints newHomo = computeHomoAndInliers(prevFrame.points, newFrame.points);
    newFrame.corners = updateCorners(prevFrame.corners, newHomo);
    newFrame.frame = frame.clone();

    Mat img = drawRectangle(frame, newFrame.corners);
    return newFrame;
}

void BookTracker::setup()
{
    computeFeaturesOnTargets();
    computeFeaturesOnFrame();

    for (int i = 0; i < targets.size(); i++)
    {
        homo.push_back(matchTargetToFrame(targets[i]));

        vector<Point2f> corners = genCornersForTarget(targets[i]);
        corners = updateCorners(corners, homo[i]);

        Mat img = drawRectangle(firstFrame.image, corners);

        namedWindow("Targets", WINDOW_NORMAL);
        resizeWindow("Targets", 600, 600);
        imshow("Targets", img);
        waitKey(0);

        this->corners.push_back(corners);
    }
        
}


void BookTracker::loop()
{
    Mat frame = firstFrame.image.clone();
    vector<frameWithPointsAndCorners> prevFrames;
    for (int i = 0; i < targets.size(); i++)
    {
        frameWithPointsAndCorners tmpFrame;
        tmpFrame.frame = firstFrame.image.clone();
        tmpFrame.points = homo[i].points;
        tmpFrame.corners = corners[i];
        prevFrames.push_back(tmpFrame);
    }
    
    Mat displayFrame;
    while (!frame.empty())
    {
        cap >> frame;
        displayFrame = frame.clone();
        for (int i = 0; i < targets.size(); i++)
        {
            prevFrames[i] = processFrame(frame, prevFrames[i]);
            displayFrame = drawRectangle(displayFrame, prevFrames[i].corners);
        }
        imshow("Video", displayFrame);
        waitKey(30);
    }
}