#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
    if (argc != 2)
        return -1;
        
    VideoCapture cap(argv[1]);
    if (cap.isOpened())
    {
        for (;;)
        {
            Mat frame;
            cap >> frame;


        }
        
    }

    return 0;
}
