#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/optflow.hpp>

#define RATIO 1.8

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
	if (argc != 3)
	{
		cout << "Usage: \n \t program <path-to-video-file> <path-to-targets> \n \t The second argument should be a valid glob string. " << endl;
		return -1;
	}

	// Get file location from command args
	String video_file = argv[1];
	vector<String> target_files;
	glob(argv[2], target_files);

	vector<Mat> targets;
	//Targets
	for (int i = 0; i < target_files.size(); i++)
		targets.push_back(imread(target_files[i]));

	//create the feature detector
	vector<vector<KeyPoint>> keyPoints_src;
	vector<Mat> descriptor_src;

	//detect and compute descriptors
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

	for (int i = 0; i < targets.size(); i++)
	{
		vector<KeyPoint> tempKey;
		Mat tempDesc;
		sift->detect(targets[i], tempKey);
		sift->compute(targets[i], tempKey, tempDesc);

		keyPoints_src.push_back(tempKey);
		descriptor_src.push_back(tempDesc);
	}

	//Object to process videos
	VideoCapture cap(video_file);

	if (cap.isOpened())
	{
		//First frame on which we should locate the objects
		Mat first_frame;
		cap >> first_frame;

		//Compute features from first frame to compare
		vector<KeyPoint> keyPoints_dst;
		Mat descriptor_dst;

		sift->detect(first_frame, keyPoints_dst);
		sift->compute(first_frame, keyPoints_dst, descriptor_dst);

		//Match Features
		BFMatcher matcher(cv::NORM_L2);
		vector<vector<DMatch>> valid_matches;

		vector<Mat> homography;
		//Computes the matches between the features extracted
		for (int i = 0; i < targets.size(); i++)
		{	
			vector<DMatch> matches;
			vector<DMatch> tmp_valid_matches;
			matcher.match(descriptor_src[i], descriptor_dst, matches);
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
				if (matches[j].distance < min_dist * RATIO)
				{
					src_kp.push_back(keyPoints_src[i][j].pt);
					dst_kp.push_back(keyPoints_dst[matches[j].trainIdx].pt);
					tmp_valid_matches.push_back(matches[j]);
				}
			valid_matches.push_back(tmp_valid_matches);
			// Use of RANSAC to discard outliers
			Mat in_mask;
			Mat tmp_homography = findHomography(src_kp, dst_kp, cv::RANSAC, 3, in_mask);
			homography.push_back(tmp_homography);
			vector<Point2f> detected_objs_pt;
			for (int j = 0; j < in_mask.rows; ++j)
				if (in_mask.at<unsigned short>(j, 0))
					detected_objs_pt.push_back(dst_kp[j]);
		}

		//Draw the matches
		for (int i = 0; i < targets.size(); i++)
		{
			Mat output;
			drawMatches(targets[i], keyPoints_src[i], first_frame, keyPoints_dst, valid_matches, output);
			namedWindow("Matches", WINDOW_NORMAL);
			resizeWindow("Matches", 600, 600);
			imshow("Matches", output);
			waitKey(0);
		}

		int index = 2;
		//Compute the traslation between src image to video
		Point3d top_left = Point3d(0, 0, 1);
		Point3d top_right = Point3d(targets[index].cols, 0, 1);
		Point3d bottom_right = Point3d(targets[index].cols, targets[index].rows, 1);
		Point3d bottom_left = Point3d(0, targets[index].rows, 1);

		vector<Mat> dst_corn;
		dst_corn.push_back(homography[index] * Mat(top_left));
		dst_corn.push_back(homography[index] * Mat(top_right));
		dst_corn.push_back(homography[index] * Mat(bottom_right));
		dst_corn.push_back(homography[index] * Mat(bottom_left));

		Point2f top_left_dst = Point2f(dst_corn[0].at<double>(0, 0), dst_corn[0].at<double>(1, 0));
		Point2f top_right_dst = Point2f(dst_corn[1].at<double>(0, 0), dst_corn[1].at<double>(1, 0));
		Point2f bttm_right_dst = Point2f(dst_corn[2].at<double>(0, 0), dst_corn[2].at<double>(1, 0));
		Point2f bttm_left_dst = Point2f(dst_corn[3].at<double>(0, 0), dst_corn[3].at<double>(1, 0));

		cout << "Top_left_dst =" << top_left_dst << endl;
		cout << "top_right_dst =" << top_right_dst << endl;
		cout << "bttm_right_dst =" << bttm_right_dst << endl;
		cout << "bttm_left_dst =" << bttm_left_dst << endl;

		//Draw the rectangle
		line(first_frame, top_left_dst, top_right_dst, Scalar(0, 255, 0), 1, LINE_AA);
		line(first_frame, top_right_dst, bttm_right_dst, Scalar(0, 255, 0), 1, LINE_AA);
		line(first_frame, bttm_right_dst, bttm_left_dst, Scalar(0, 255, 0), 1, LINE_AA);
		line(first_frame, bttm_left_dst, top_left_dst, Scalar(0, 255, 0), 1, LINE_AA);
		namedWindow("Targets", WINDOW_NORMAL);
		resizeWindow("Targets", 600, 600);
		imshow("Targets", first_frame);
		waitKey(0);

		Mat previous;
		cv::cvtColor(first_frame, previous, cv::COLOR_BGR2GRAY);

		int condition = 1;
		while (condition)
		{
			Mat frame;
			cap >> frame;

			if (frame.empty())
				condition = 0;
			else
			{
				vector<Point2f> prev_corn_vec;
				vector<Point2f> next_corn_vec;
				vector<uchar> status;
				vector<float> err;
				TermCriteria term = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

				prev_corn_vec.push_back(top_left_dst);
				prev_corn_vec.push_back(top_right_dst);
				prev_corn_vec.push_back(bttm_right_dst);
				prev_corn_vec.push_back(bttm_left_dst);

				cvtColor(first_frame, previous, cv::COLOR_BGR2GRAY);

				calcOpticalFlowPyrLK(previous, frame, prev_corn_vec, next_corn_vec, status, err,
									 Size(21, 21), 3, term, 0);

				// I am preparing for the next image
				swap(frame, previous);
				top_left_dst = prev_corn_vec[0];
				top_right_dst = prev_corn_vec[1];
				bttm_right_dst = prev_corn_vec[2];
				bttm_left_dst = prev_corn_vec[3];

				line(frame, top_left_dst, top_right_dst, Scalar(0, 255, 0), 1, LINE_AA);
				line(frame, top_right_dst, bttm_right_dst, Scalar(0, 255, 0), 1, LINE_AA);
				line(frame, bttm_right_dst, bttm_left_dst, Scalar(0, 255, 0), 1, LINE_AA);
				line(frame, bttm_left_dst, top_left_dst, Scalar(0, 255, 0), 1, LINE_AA);
				imshow("TMP2", frame);
				waitKey(0);
			}
		}
	}

	return 0;
}
