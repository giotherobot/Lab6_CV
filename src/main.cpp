#include <BookTracker.h>

using namespace cv;
using namespace std;


int main(int argc, char const *argv[])
{
    if (argc != 3)
	{
		cout << "Usage: \n \t lab6 <path-to-video-file> <path-to-targets> \n \t The second argument should be a valid glob string. " << endl;
		return -1;
	}

	// Get file location from command args
	String video_file = argv[1];
	vector<String> target_files;
	glob(argv[2], target_files);

    BookTracker tracker = BookTracker(5, 2000, 50000);

    tracker.loadTargets(target_files);
    tracker.loadVideo(video_file);

    tracker.setup();
    
    tracker.loop();

    return 0;
}
