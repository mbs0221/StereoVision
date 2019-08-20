#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

CvCapture *pCapture1, *pCapture2;
IplImage * img1, *img2, *pstacked;

IplImage* stack_imgs(IplImage* img1, IplImage* img2)
{
	IplImage* stacked = cvCreateImage(cvSize(img1->width + img2->width, MAX(img1->height, img2->height)), IPL_DEPTH_8U, 3);

	cvZero(stacked);
	cvSetImageROI(stacked, cvRect(0, 0, img1->width, img1->height));
	cvAdd(img1, stacked, stacked, NULL);
	cvSetImageROI(stacked, cvRect(img1->width, 0, img2->width, img2->height));
	cvAdd(img2, stacked, stacked, NULL);
	cvResetImageROI(stacked);

	return stacked;
}

int i = 1;

void mouseHandler(int event, int x, int y, int flags, void* param) {
	char buf[32];
	switch (event)
	{
	case CV_EVENT_FLAG_LBUTTON:
		sprintf(buf, "left%02d.png", i);
		cvSaveImage(buf, img1);
		printf("save:%s\n", buf);
		sprintf(buf, "right%02d.png", i);
		cvSaveImage(buf, img2);
		printf("save:%s\n", buf);
		i++;
		break;
	case CV_EVENT_FLAG_RBUTTON:
		i = 1;
		break;
	default:
		break;
	}
}

int main(int argc, char *argv[])
{
	pCapture1 = cvCreateCameraCapture(0);
	pCapture2 = cvCreateCameraCapture(1);
/*
	cvSetCaptureProperty(pCapture1, CV_CAP_PROP_FRAME_WIDTH, 320);
	cvSetCaptureProperty(pCapture1, CV_CAP_PROP_FRAME_HEIGHT, 240);
	cvSetCaptureProperty(pCapture2, CV_CAP_PROP_FRAME_WIDTH, 320);
	cvSetCaptureProperty(pCapture2, CV_CAP_PROP_FRAME_HEIGHT, 240);*/

	cout << "camera" << endl;
	cvNamedWindow("camera", 1);
	cvSetMouseCallback("camera", mouseHandler, 0);

	while (1) {
		img1 = cvQueryFrame(pCapture1);
		if (!img1) {
			printf("img1 is null\n");
			break;
		}
		img2 = cvQueryFrame(pCapture2);
		if (!img2) {
			printf("img2 is null\n");
			break;
		}
		pstacked = stack_imgs(img1, img2);
		if (!pstacked) {
			printf("stacked img is null\n");
			break;
		}

		cvShowImage("camera", pstacked);
		cvReleaseImage(&pstacked);

		int c = cvWaitKey(33);
		if (c == 'n')break;
	}
	cvWaitKey(0);
}
