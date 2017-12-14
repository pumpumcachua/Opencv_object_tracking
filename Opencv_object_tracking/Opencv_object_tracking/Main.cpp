
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>
#include <algorithm>
using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< ostringstream & >( \
( ostringstream() << dec << x ) ).str()
Ptr<Tracker> TrackerAlgorithm(string trackerType)
{
	Ptr<Tracker> tracker;
	if (trackerType == "BOOSTING")
		tracker = TrackerBoosting::create();
	if (trackerType == "MIL")
		tracker = TrackerMIL::create();
	if (trackerType == "KCF")
		tracker = TrackerKCF::create();
	if (trackerType == "TLD")
		tracker = TrackerTLD::create();
	if (trackerType == "MEDIANFLOW")
		tracker = TrackerMedianFlow::create();
	if (trackerType == "GOTURN")
		tracker = TrackerGOTURN::create();
	return tracker;
}

/*
int main(int argc, char **argv)
{
	
	// List of tracker types in OpenCV 3.2
	// NOTE : GOTURN implementation is buggy and does not work.
	string trackerTypes[6] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN" };
	// vector <string> trackerTypes(types, end(types));

	// Create a tracker
	string trackerType = trackerTypes[2];

	Ptr<Tracker> tracker;
	{
		if (trackerType == "BOOSTING")
			tracker = TrackerBoosting::create();
		if (trackerType == "MIL")
			tracker = TrackerMIL::create();
		if (trackerType == "KCF")
			tracker = TrackerKCF::create();
		if (trackerType == "TLD")
			tracker = TrackerTLD::create();
		if (trackerType == "MEDIANFLOW")
			tracker = TrackerMedianFlow::create();
		if (trackerType == "GOTURN")
			tracker = TrackerGOTURN::create();
	}

	// Read video
	VideoCapture video;
	//video.open("C:\\Users\\user\\source\\repos\\Object-tracking\\x64\\Release\\videos\\chaplin.mp4");
	//video.open("C:\\Users\\user\\source\\repos\\Object-tracking\\x64\\Release\\videos\\ball_tracking_example.mp4");
	video.open("C:\\Users\\user\\source\\repos\\Object-tracking\\x64\\Release\\videos\\1.mp4");
	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return 1;

	}

	// Read first frame
	Mat frame;
	
	bool ok = video.read(frame);

	// Define initial boundibg box
	Rect2d bbox(287, 23, 86, 320);

	// Uncomment the line below to select a different bounding box
	bbox = selectROI(frame, false);

	// Display bounding box.
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	imshow("Tracking", frame);

	tracker->init(frame, bbox);

	while (video.read(frame))
	{

		// Start timer
		double timer = (double)getTickCount();

		// Update the tracking result
		bool ok = tracker->update(frame, bbox);

		// Calculate Frames per second (FPS)
		float fps = getTickFrequency() / ((double)getTickCount() - timer);


		if (ok)
		{
			// Tracking success : Draw the tracked object
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		}
		else
		{
			// Tracking failure detected.
			putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		}

		// Display tracker type on frame
		putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Display FPS on frame
		putText(frame, "FPS : " + SSTR(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Display frame.
		imshow("Tracking", frame);

		// Exit if ESC pressed.
		int k = waitKey(1);
		if (k == 27)
		{
			break;
		}

	}


	
	
}
*/

int main(int argc, char** argv) {
	
	 // show help
	  if (argc<2) {
		  cout <<
			  " Usage: Object-tracking <video_path> [algorithm]\n"
			  " Algorithm: BOOSTING MIL KCF TLD MEDIANFLOW"
			  " examples:\n"
			  " Object-tracking Bolt/img/%04d.jpg\n"
			  " Object-tracking faceocc2.webm MEDIANFLOW\n"
			  << endl;;
	    return -1;
		}
	
	// set the default tracking algorithm (BOOSTING,MIL,KCF,TLD,MEDIANFLOW) ( GOTURN -bugged)
	string trackingAlg = "KCF";
	string AlgorithmList[6] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW" };
	// set the tracking algorithm from parameter
	if (argc>2)
	  trackingAlg = argv[2];
	
	//check if the algorithm is existed
	bool exists = find(begin(AlgorithmList), end(AlgorithmList), trackingAlg) != end(AlgorithmList);
	if (!exists)
	{
		cout << "Invalid tracking algorithm " << endl;
		return -1;
	}
	// create the tracker
	MultiTracker trackers;
	
	// container of the tracked objects
	vector<Rect2d> objects;
	
	// set input video
	string video = argv[1];
	//bouncingBall.avi chaplin.mp4 ball_tracking_example
	VideoCapture cap(video);
	if (!cap.isOpened())
	{
		cout << "Can not open the video: " << video << endl;
		return -1;
	}
	Mat frame;
	
	// get bounding box
	while (waitKey(0) != 27)
	{
		 cout << " Press any key to move to next frame!" << endl 
			 << " Press Esc to choose starting frame" << endl;
		 cap.read(frame);
		 imshow("tracker",frame);
	}
	vector<Rect> ROIs;
	selectROIs("tracker", frame, ROIs);
	
	//quit when the tracked object(s) is not provided
	if (ROIs.size()<1)
	  return 0;
	
	// initialize the tracker
	vector<Ptr<Tracker> > algorithms;
	vector<deque<Point>> centerQueues;
	vector<Vec3b> colors;
	Mat hsvframe;
	cvtColor(frame, hsvframe, CV_BGR2HSV);
	vector<Vec3b> hsvVector;
	for (size_t i = 0; i < ROIs.size(); i++)
	{
	    algorithms.push_back(TrackerAlgorithm(trackingAlg));
	    objects.push_back(ROIs[i]);
		 deque<Point> tmpQueue;
		 Point center = (ROIs[i].br() + ROIs[i].tl())*0.5;
		 tmpQueue.push_back(center);
		 centerQueues.push_back(tmpQueue);
		 uchar blue = rand() % 255;
		 uchar green = rand() % 255;
		 uchar red=rand() % 255;
		 Vec3b tmpcolor(blue, green, red);
		 colors.push_back(tmpcolor);
		 Vec3b hsvTmp = hsvframe.at<Vec3b>(center);
		 hsvVector.push_back(hsvTmp);
	 }
	
	trackers.add(algorithms, frame, objects);
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;
	// do the tracking
	cout << "Start the tracking process, press ESC to quit." <<endl;
	while (cap.read(frame))
	{
		Mat mask;
		Vec3b hsvPoint = hsvVector.back();
		int Hue = hsvPoint.val[0];
		int H_range = (Hue >= 128) ? 255 - Hue : Hue;
		int Saturation = hsvPoint.val[1];
		int S_range = (Saturation >= 128) ? 255 - Saturation : Saturation;
		int Value = hsvPoint.val[2];
		int V_range = (Value >= 128) ? 255 - Value : Value;
		cvtColor(frame, hsvframe, CV_BGR2HSV);
		inRange(hsvframe, Scalar(Hue-H_range/2, Saturation- S_range/2, Value - V_range/2), Scalar(Hue + H_range/2, Saturation + S_range / 2, Value + V_range / 2),mask);
		if (mask.empty())
			return 0;
		erode(mask, mask,0,Point(-1,-1),2);
		dilate(mask, mask, 0, Point(-1, -1), 2);

		Mat clone = mask.clone();
		vector<vector<Point>> cnts;
		findContours(clone, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE,Point(0,0));
		for (int i = 0; i< cnts.size(); i++) // iterate through each contour. 
		{
			double a = contourArea(cnts[i], false);  //  Find the area of contour
			if (a>largest_area) {
				largest_area = a;
				largest_contour_index = i;                //Store the index of largest contour
				bounding_rect = boundingRect(cnts[i]); // Find the bounding rectangle for biggest contour
			}
		}
		rectangle(frame, bounding_rect, Scalar(0, 255, 0), 1, 8, 0);
		imshow("Threshold Image", mask);
		// stop the program if no more images
		if (frame.rows == 0 || frame.cols == 0)
			break;

		//update the tracking result
		trackers.update(frame);

		// draw the tracked object
		for (unsigned i = 0; i < trackers.getObjects().size(); i++)
		{
			Rect2d Box = trackers.getObjects()[i];
			rectangle(frame, Box, Scalar(255, 0, 0), 2, 1);
			centerQueues[i].push_front((Box.br() + Box.tl())*0.5);

			for (int j = 0; j < centerQueues[i].size() - 1 && j < 200; j++)
			{
				line(frame, centerQueues[i][j], centerQueues[i][j + 1], colors[i], 1, 8, 0);
				//cout << centerQueues[i][j].x << " " << centerQueues[i][j].y << endl << colors[i] << endl;
			}
		}
		// Display tracker type on frame
		putText(frame, trackingAlg + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// show image with the tracked object
		imshow("tracker", frame);

		//quit on ESC button
		if (waitKey(1) == 27)break;

	}
	return 0;
}




/*
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
#include<iostream>
#ifdef WINDOWS
#include<conio.h>           // it may be necessary to change or remove this line if not using Windows
#endif

#include "Blob.h"

// global variables ///////////////////////////////////////////////////////////////////////////////
const Scalar SCALAR_BLACK = Scalar(0.0, 0.0, 0.0);
const Scalar SCALAR_WHITE = Scalar(255.0, 255.0, 255.0);
const Scalar SCALAR_BLUE = Scalar(255.0, 0.0, 0.0);
const Scalar SCALAR_GREEN = Scalar(0.0, 200.0, 0.0);
const Scalar SCALAR_RED = Scalar(0.0, 0.0, 255.0);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

	VideoCapture capVideo;

	Mat imgFrame1;
	Mat imgFrame2;

	capVideo.open("videos\\1.mp4");

	if (!capVideo.isOpened()) {                                                 // if unable to open video file
		cout << "\nerror reading video file" << endl << endl;      // show error message
#ifdef WINDOWS
		_getch();                    // it may be necessary to change or remove this line if not using Windows
#endif
		return(0);                                                              // and exit program
	}

	if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
		cout << "\nerror: video file must have at least two frames";
#ifdef WINDOWS
		_getch();
#endif
		return(0);
	}

	capVideo.read(imgFrame1);
	capVideo.read(imgFrame2);

	char chCheckForEscKey = 0;

	while (capVideo.isOpened() && chCheckForEscKey != 27) {

		vector<Blob> blobs;

		Mat imgFrame1Copy = imgFrame1.clone();
		Mat imgFrame2Copy = imgFrame2.clone();

		Mat imgDifference;
		Mat imgThresh;

		cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
		cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

		GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 0);
		GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 0);

		absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

		threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

		imshow("imgThresh", imgThresh);

		Mat structuringElement3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat structuringElement5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));
		Mat structuringElement7x7 = getStructuringElement(MORPH_RECT, Size(7, 7));
		Mat structuringElement9x9 = getStructuringElement(MORPH_RECT, Size(9, 9));

		dilate(imgThresh, imgThresh, structuringElement5x5);
		dilate(imgThresh, imgThresh, structuringElement5x5);
		erode(imgThresh, imgThresh, structuringElement5x5);

		Mat imgThreshCopy = imgThresh.clone();

		vector<vector<Point> > contours;

		findContours(imgThreshCopy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		Mat imgContours(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		drawContours(imgContours, contours, -1, SCALAR_WHITE, -1);

		imshow("imgContours", imgContours);

		vector<vector<Point> > convexHulls(contours.size());

		for (unsigned int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convexHulls[i]);
		}

		for (auto &convexHull : convexHulls) {
			Blob possibleBlob(convexHull);

			if (possibleBlob.boundingRect.area() > 100 &&
				possibleBlob.dblAspectRatio >= 0.2 &&
				possibleBlob.dblAspectRatio <= 1.2 &&
				possibleBlob.boundingRect.width > 15 &&
				possibleBlob.boundingRect.height > 20 &&
				possibleBlob.dblDiagonalSize > 30.0) {
				blobs.push_back(possibleBlob);
			}
		}

		Mat imgConvexHulls(imgThresh.size(), CV_8UC3, SCALAR_BLACK);

		convexHulls.clear();

		for (auto &blob : blobs) {
			convexHulls.push_back(blob.contour);
		}

		drawContours(imgConvexHulls, convexHulls, -1, SCALAR_WHITE, -1);

		imshow("imgConvexHulls", imgConvexHulls);

		imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

		for (auto &blob : blobs) {                                                  // for each blob
			rectangle(imgFrame2Copy, blob.boundingRect, SCALAR_RED, 2);             // draw a red box around the blob
			circle(imgFrame2Copy, blob.centerPosition, 3, SCALAR_GREEN, -1);        // draw a filled-in green circle at the center
		}

		imshow("imgFrame2Copy", imgFrame2Copy);

		// now we prepare for the next iteration

		imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

		if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {       // if there is at least one more frame
			capVideo.read(imgFrame2);                            // read it
		}
		else {                                                  // else
			cout << "end of video\n";                      // show end of video message
			break;                                              // and jump out of while loop
		}

		chCheckForEscKey = waitKey(1);      // get key press in case user pressed esc

	}

	if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
		waitKey(0);                         // hold the windows open to allow the "end of video" message to show
	}
	// note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

	return(0);
}
*/