#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/optflow.hpp>

using namespace cv;
using namespace std;

struct imageWithFeatures
{
    Mat image;
    vector<KeyPoint> keypoints;
    Mat descriptors;
};

struct homoWithPoints
{
    Mat homography;
    vector<Point2f> points;
};

struct pointsWithStatus
{
    vector<Point2f> points;
    vector<uchar> status;
};

struct frameWithPointsAndCorners
{
    Mat frame;
    pointsWithStatus points;
    vector<Point2f> corners;
};

class BookTracker
{
private:
    float homoExcludeRatio = 3;
    int nfeaturesTargets = 5000;
    int nfeaturesFrame = 20000;


    VideoCapture cap;
    vector<imageWithFeatures> targets;
    imageWithFeatures firstFrame;
    vector<homoWithPoints> homo;

    vector<vector<Point2f>> corners;
    vector<Scalar> colors;


public:
    BookTracker(){};
    BookTracker(float homoExcludeRatio, int nfeaturesTargets, int nfeaturesFrame)
    {
        this->homoExcludeRatio = homoExcludeRatio;
        this->nfeaturesTargets = nfeaturesTargets;
        this->nfeaturesFrame = nfeaturesFrame;

        colors.push_back(Scalar(0, 0, 255));
        colors.push_back(Scalar(0, 255, 255));
        colors.push_back(Scalar(0, 255, 0));
        colors.push_back(Scalar(255, 0, 0));
        colors.push_back(Scalar(255, 0, 255));
        colors.push_back(Scalar(255, 255, 0));
    }

    void loadTargets(vector<String> target_files);
    bool loadVideo(String video_file);

    void computeFeaturesOnTargets();
    void computeFeaturesOnFrame();
    void drawTrackedFeatures(Mat img, vector<Point2f> features, Scalar color);

    homoWithPoints matchTargetToFrame(imageWithFeatures target);

    vector<Point2f> genCornersForTarget(imageWithFeatures target);
    vector<Point2f> updateCorners(vector<Point2f> corners, homoWithPoints homo);
    Mat drawRectangle(Mat src, vector<Point2f> corners, Scalar color);

    homoWithPoints computeHomoAndInliers(pointsWithStatus src_kp, pointsWithStatus dst_kp);
    homoWithPoints computeHomoAndInliers(vector<Point2f> src_kp, vector<Point2f> dst_kp);

    pointsWithStatus computeOptFlow(Mat prevFrame, Mat frame, vector<Point2f> prevKPoints);

    frameWithPointsAndCorners processFrame(Mat frame, frameWithPointsAndCorners prevFrame, Scalar color);

    void setup();
    void loop();

};
