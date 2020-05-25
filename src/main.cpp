#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ccalib.hpp>

#define VIDEO "./data/video.mov"
#define TARGET "./data/objects/obj2.png"
#define RATIO 1.5

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
	//Target
	Mat target = imread(TARGET);

	//create the feature detector
	vector<KeyPoint> keyPoints_src;
	Mat descriptor_src;

	//detect and compute descriptors
	Ptr<ORB> orb = ORB::create();
	orb->detect(target, keyPoints_src);
	orb->compute(target, keyPoints_src, descriptor_src);

    //Object to process videos
    VideoCapture cap(VIDEO);

    int condition = 0;
    if (cap.isOpened())
    {
    	//First frame on which we should locate the objects
    	Mat first_frame;
    	cap >> first_frame;

    	//Compute features from first frame to compare
		vector<KeyPoint> keyPoints_dst;
		Mat descriptor_dst;

		orb->detect(first_frame, keyPoints_dst);
		orb->compute(first_frame, keyPoints_dst, descriptor_dst);

		//Match Features
		BFMatcher matcher(cv::NORM_L2);
		vector<DMatch> matches;
		vector<DMatch> valid_matches;

		//Computes the matches between the features extracted
		matcher.match(descriptor_src, descriptor_dst, matches);

		//Keep Only matches under a certain threashold
		vector<Point2f> src_kp;
		vector<Point2f> dst_kp;

		//Finding min dist
		float min_dist = matches[0].distance;
		for ( int j = 1 ; j < matches.size(); ++j )
			if ( min_dist > matches[j].distance)
				min_dist = matches[j].distance;

		//Refine the matches
		for ( int j = 0 ; j < matches.size(); ++j )
			if ( matches[j].distance < min_dist * RATIO )
			{
					src_kp.push_back(keyPoints_src[j].pt);
					dst_kp.push_back(keyPoints_dst[matches[j].trainIdx].pt);
					valid_matches.push_back(matches[j]);
			}

		// Use of RANSAC to discard outliers
		Mat in_mask;
		Mat homography = findHomography(src_kp, dst_kp, cv::RANSAC, 3, in_mask);
		vector<Point2f> detected_objs_pt;

		for ( int j = 0 ; j < in_mask.rows ; ++j)
			if ( in_mask.at<unsigned short>(j, 0) )
				detected_objs_pt.push_back(dst_kp[j]);

		//For now it does nothing
        while ( condition )
        {
            Mat frame;
            cap >> frame;

        }
        
        cout << "The detected object points are : " << endl;
        cout << detected_objs_pt  << endl;

        //Draw the matches
        Mat output;
        drawMatches(target, keyPoints_src, first_frame , keyPoints_dst, valid_matches, output);
        resize(output, output, Size(output.cols/2, output.rows/2));
        imshow("TMP5", output);

        waitKey(0);

        //Compute the traslation between src image to video
        Point3d top_left = Point3d(0,0, 1);
        Point3d top_right = Point3d(0, target.cols, 1);
        Point3d bottom_right = Point3d(target.rows, target.cols, 1);
        Point3d bottom_left = Point3d(target.rows, 0, 1);

        vector<Mat> dst_corn;
        dst_corn.push_back(homography * Mat(top_left));
        cout << "It works #1" << endl;
        dst_corn.push_back(homography * Mat(top_right));
        cout << "It works #2" << endl;
        dst_corn.push_back(homography * Mat(bottom_right));
        cout << "It works #3" << endl;
        dst_corn.push_back(homography * Mat(bottom_left));
        cout << "It works #4" << endl;

        cout << "Before Top_left_dst =" << dst_corn[0] << endl;
		cout << "Top_right_dst =" << dst_corn[1] << endl;
		cout << "Before Bottom_i_dst =" << dst_corn[2] << endl;
		cout << "Before Bottom_Left_dst =" << dst_corn[3] << endl;

		Point2f top_left_dst = Point2f(dst_corn[0].at<double>(0, 0), dst_corn[0].at<double>(1, 0));
		Point2f top_right_dst = Point2f(dst_corn[1].at<double>(0, 0), dst_corn[1].at<double>(1, 0));
		Point2f bttm_right_dst = Point2f(dst_corn[2].at<double>(0, 0), dst_corn[2].at<double>(1, 0));
		Point2f bttm_left_dst = Point2f(dst_corn[3].at<double>(0, 0), dst_corn[3].at<double>(1, 0));
		cout << "Top_left_dst =" << top_left_dst << endl;
		cout << "Top_right_dst =" << top_right_dst << endl;
		cout << "Bottom_right_dst =" << bttm_right_dst << endl;
		cout << "Bottom_left_dst =" << bttm_left_dst << endl;

		//Draw the rectangle
        line(first_frame, top_left_dst, top_right_dst, Scalar(0, 255, 0), 1, LINE_AA );
        line(first_frame, top_right_dst, bttm_right_dst, Scalar(0, 255, 0), 1, LINE_AA );
        line(first_frame, bttm_right_dst, bttm_left_dst, Scalar(0, 255, 0), 1, LINE_AA );
        line(first_frame, bttm_left_dst, top_left_dst, Scalar(0, 255, 0), 1, LINE_AA );
        imshow("TMP", first_frame);
        imshow("TMP1", first_frame);
        imshow("TMP2", first_frame);
        waitKey(0);

    }

    return 0;
}
