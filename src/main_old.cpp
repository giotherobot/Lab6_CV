#include <BookTracker.h>

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
	vector<vector<KeyPoint>> target_kp;
	vector<Mat> target_desc;

	//detect and compute descriptors
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(500);

	for (int i = 0; i < targets.size(); i++)
	{
		vector<KeyPoint> tempKey;
		Mat tempDesc;
		sift->detect(targets[i], tempKey);
		sift->compute(targets[i], tempKey, tempDesc);

		target_kp.push_back(tempKey);
		target_desc.push_back(tempDesc);
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

		Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(10000);

		sift->detect(first_frame, keyPoints_dst);
		sift->compute(first_frame, keyPoints_dst, descriptor_dst);

		//Match Features
		BFMatcher matcher(cv::NORM_L2);
		vector<vector<DMatch>> valid_matches;

		vector<Mat> homography;
		vector<vector<Point2f>> detected_points;

		//Computes the matches between the features extracted
		for (int i = 0; i < targets.size(); i++)
		{
			vector<DMatch> matches;
			vector<DMatch> tmp_valid_matches;
			matcher.match(target_desc[i], descriptor_dst, matches);

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
				if (matches[j].distance < min_dist * 3)
				{
					src_kp.push_back(target_kp[i][j].pt);
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
			detected_points.push_back(detected_objs_pt);
		}

		//Draw the matches
		for (int i = 0; i < targets.size(); i++)
		{
			Mat output;
			drawMatches(targets[i], target_kp[i], first_frame, keyPoints_dst, valid_matches[i], output);
			namedWindow("Matches", WINDOW_NORMAL);
			resizeWindow("Matches", 600, 600);
			imshow("Matches", output);
			waitKey(0);
		}

		vector<vector<Point2f>> img_corners;
		for (int index = 0; index < 4; index++)
		{
			//Compute the traslation between src image to video
			vector<Point2f> corners;
			corners.push_back(Point2f(0, 0));
			corners.push_back(Point2f((float)targets[index].cols, 0));
			corners.push_back(Point2f((float)targets[index].cols, (float)targets[index].rows));
			corners.push_back(Point2f(0, (float)targets[index].rows));

			vector<Point2f> dst_corn;
			perspectiveTransform(corners, dst_corn, homography[index]);

			// Draw the rectangle
			line(first_frame, dst_corn[0], dst_corn[1], Scalar(0, 255, 0), 4);
			line(first_frame, dst_corn[1], dst_corn[2], Scalar(0, 255, 0), 4);
			line(first_frame, dst_corn[2], dst_corn[3], Scalar(0, 255, 0), 4);
			line(first_frame, dst_corn[3], dst_corn[0], Scalar(0, 255, 0), 4);

			for (int i = 0; i < detected_points[index].size(); i++)
			{
				circle(first_frame, detected_points[index][i], 3, Scalar(0, 0, 255));
			}
			img_corners.push_back(dst_corn);

			namedWindow("Targets", WINDOW_NORMAL);
			resizeWindow("Targets", 600, 600);
			imshow("Targets", first_frame);
			waitKey(0);
		}

		Mat previous, frame, gray_frame;
		cv::cvtColor(first_frame, previous, cv::COLOR_BGR2GRAY);
		frame = first_frame.clone();
		vector<Point2f> prev_points = detected_points[1];
		vector<Point2f> next_points;
		vector<Point2f> prev_corners = img_corners[1];
		vector<Point2f> next_corners;

		while (!frame.empty())
		{
			cap >> frame;
			vector<uchar> status;
			vector<float> err;
			TermCriteria term = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);

			cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

			calcOpticalFlowPyrLK(previous, gray_frame, prev_points, next_points, status, err,
								 Size(21, 21), 3, term, 0);

			vector<Point2f> new_pnt;
			for (size_t i = 0; i < prev_points.size(); i++)
				if (status[i] == 1)
					new_pnt.push_back(next_points[i]);

			Mat mask;
			Mat homo = findHomography(prev_points, new_pnt, cv::RANSAC, 3, mask);

			perspectiveTransform(prev_corners, next_corners, homo);

			// Draw the rectangle
			line(frame, next_corners[0], next_corners[1], Scalar(0, 255, 0), 4);
			line(frame, next_corners[1], next_corners[2], Scalar(0, 255, 0), 4);
			line(frame, next_corners[2], next_corners[3], Scalar(0, 255, 0), 4);
			line(frame, next_corners[3], next_corners[0], Scalar(0, 255, 0), 4);

			imshow("TMP2", frame);
			waitKey(30);

			// I am preparing for the next image
			prev_points = new_pnt;
			prev_corners = next_corners;

			previous = gray_frame.clone();
		}
	}

	return 0;
}
