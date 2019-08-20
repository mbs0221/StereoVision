#include <cv.h> 
#include <cxcore.h> 
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>

using namespace cv;
using namespace std;

static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

static bool runCalibration(vector<vector<Point2f> > imagePoints,
						   Size imageSize, Size boardSize,
						   float squareSize, float aspectRatio,
						   int flags, Mat& cameraMatrix, Mat& distCoeffs,
						   vector<Mat>& rvecs, vector<Mat>& tvecs)
{
	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (flags & CV_CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = aspectRatio;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	// 计算棋盘角点位置
	vector<vector<Point3f> > objectPoints(1);
	objectPoints[0].resize(0);
	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			objectPoints[0].push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));

	objectPoints.resize(imagePoints.size(), objectPoints[0]);
	// 单摄像头标定
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, flags | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);
	printf("RMS error reported by calibrateCamera: %g\n", rms);

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	return ok;
}

// 标定单摄像头
static void SingleCalib(string inputFilename, vector<vector<Point2f> >& imagePoints, Mat& cameraMatrix, Mat& distCoeffs, int nframes, Size boardSize, Size imageSize, float squareSize)
{
	float aspectRatio = 1.0f;
	int flags = 0;
	bool undistortImage = false;
	vector<string> imageList;
	vector<Mat> rvecs, tvecs;
	// 读取图片列表
	readStringList(inputFilename, imageList);

	if (!imageList.empty())
		nframes = (int)imageList.size();

	// 循环读取图片
	for (int i = 0; i < nframes; i++)
	{
		Mat view, viewGray;
		view = imread(imageList[i], 1);
		// 转灰度图
		vector<Point2f> pointbuf;
		cvtColor(view, viewGray, COLOR_BGR2GRAY, 1);
		// 寻找角点
		bool found;
		found = findChessboardCorners(view, boardSize, pointbuf, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		// 提升查找到角点的精确度
		if (found) cornerSubPix(viewGray, pointbuf, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		// 添加角点
		imagePoints.push_back(pointbuf);
		// 绘制角点
		if (found)
			drawChessboardCorners(view, boardSize, Mat(pointbuf), found);
		// 校正图像
		if (undistortImage)
		{
			Mat temp = view.clone();
			undistort(temp, view, cameraMatrix, distCoeffs);
		}
		// 显示图像
		// imshow("Image View", view);
		waitKey(33);
	}
	if (imagePoints.size() > 0)
	{
		runCalibration(imagePoints, imageSize, boardSize, squareSize, aspectRatio, flags, cameraMatrix, distCoeffs, rvecs, tvecs);
	}
}

// 视差图，视差灰度图
Mat disp, disp8, xyz;

// 靠靠靠
Mat mergeMat(Mat a, Mat b) {
	// 靠靠靠靠靠
	int totalCols = a.cols + b.cols;
	// 靠靠靠靠
	Mat mergedDescriptors(a.rows, totalCols, a.type());
	// 靠a靠靠靠靠
	Mat submat = mergedDescriptors.colRange(0, a.cols);
	a.copyTo(submat);
	// 靠b靠靠靠靠
	submat = mergedDescriptors.colRange(a.cols, totalCols);
	b.copyTo(submat);
	return mergedDescriptors;
}

// 靠靠靠
void mouseHandler(int event, int x, int y, int flags, void* param)
{
	Vec3f point;
	switch (event)
	{
		case CV_EVENT_FLAG_LBUTTON:
			point = xyz.at<Vec3f>(y, x);
			printf("xyz:%f %f %f\n", point[0], point[1], point[2]);
			break;
		case CV_EVENT_FLAG_RBUTTON:
			break;
		default:
			break;
	}
}

// 保存三维点云的信息
static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

// 矫正图像，计算视差
static void RectifyComputeDisparity(const char* point_cloud_filename, Mat& Q, Mat& mxl, Mat& myl, Mat& mxr, Mat& myr)
{
	// 摄像头配置
	printf("摄像头配置\n");
	VideoCapture capL(0), capR(1);
	CvSize imSize = cvSize(640, 480);
	// 定义两个图像对象
	Mat FrameL, FrameR;
	// 定义四个矩阵用于存放校正前后的灰度图像
	Mat img1, img2, img1r, img2r, pair;
	// SGBM
	printf("设置SGBM参数\n");
	cv::StereoSGBM sgbm;
	int SADWindowSize = 9;
	int cn = 3;// RGB三通道
	int numberOfDisparities = 32;
	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm.P1 = 4 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32 * cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = numberOfDisparities;
	sgbm.uniquenessRatio = 10;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	// 显示窗口
	namedWindow("disparity");
	// namedWindow("rectified", 1);
	cvSetMouseCallback("disparity", mouseHandler, 0);
	printf("循环读取左右相机的图像\n");
	// 循环读取左右相机的图像
	while (capL.isOpened() && capR.isOpened())
	{
		// 读取摄像头
		capL.read(FrameL);
		capR.read(FrameR);
		// 使用用前面计算的mx和my立体校正图片，两个相机的图片处于同一个相机平面且行对准
		cvtColor(FrameL, img1, COLOR_BGR2GRAY);
		cvtColor(FrameR, img2, COLOR_BGR2GRAY);
		remap(img1, img1r, mxl, myl, CV_INTER_LINEAR);
		remap(img2, img2r, mxr, myr, CV_INTER_LINEAR);
		// SGBM匹配获取视差图
		sgbm(img2r, img1r, disp);
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.0));
		// 使用Q矩阵将视差图投影到三维坐标
		reprojectImageTo3D(disp, xyz, Q, true);
		// 保存三维点云数据
		// saveXYZ(point_cloud_filename, xyz);
		// TODO: 将立体矫正后的左右图像合并
		pair = mergeMat(img1r, img2r);
		// TODO: 在一定距离间隔处绘制水平直线		
		// 显示视差图和矫正后的左右摄像头图像
		imshow("disparity", disp8);
		imshow("rectified", pair);
		char c = cvWaitKey(33);
		if (c == 'Y')break;
	}
}

int main(int argc, char* argv[])
{
	const char* inputFilenameL = "left.txt";
	const char* inputFilenameR = "right.txt";
	const char* point_cloud_filename = "point_cloud.txt";
	int i, j, k, nimages = 15;// 标定板图片数量
	float squareSize = 2.5;// 方格大小2.5cm
	bool useCalibrated = false;
	Size boardSize(7, 6);// 角点尺寸
	Size imageSize(640, 480);// 图像大小
	vector<vector<Point3f> > objectPoints;// 棋盘角点
	vector<vector<Point2f> > imagePointsL, imagePointsR;//左、右相机图片中的角点
	Mat cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR;//左右相机的内参数矩阵和畸变系数向量
	cout << "单摄像头标定" << endl;
	SingleCalib(inputFilenameL, imagePointsL, cameraMatrixL, distCoeffsL, nimages, boardSize, imageSize, squareSize);
	SingleCalib(inputFilenameR, imagePointsR, cameraMatrixR, distCoeffsR, nimages, boardSize, imageSize, squareSize);
	cout << "立体标定" << endl;
	Mat R, T, E, F;//旋转矩阵、平移矩阵、本征矩阵、基础矩阵
	objectPoints.resize(nimages);
	for (int i = 0; i < nimages; i++)
	{
		for (int j = 0; j < boardSize.height; j++)
			for (int k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}
	double rms = stereoCalibrate(objectPoints, imagePointsL, imagePointsR,
								 cameraMatrixL, distCoeffsL,
								 cameraMatrixR, distCoeffsR,
								 imageSize, R, T, E, F,
								 TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 100, 1e-5),
								 CV_CALIB_FIX_ASPECT_RATIO +
								 CV_CALIB_ZERO_TANGENT_DIST +
								 CV_CALIB_SAME_FOCAL_LENGTH +
								 CV_CALIB_RATIONAL_MODEL +
								 CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5);
	cout << "done with RMS error=" << rms << endl;
	cout << "立体校正" << endl;
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];
	stereoRectify(cameraMatrixL, distCoeffsL,
				  cameraMatrixR, distCoeffsR,
				  imageSize, R, T, R1, R2, P1, P2, Q,
				  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
	if (useCalibrated)
	{
		// IF BY CALIBRATED (BOUGUET'S METHOD)
		// 已经在stereoRectify时完成了
	}
	else
	{
		// HARTLEY'S METHOD
		cout << "使用HARTLEY'S METHOD计算矫正变换矩阵" << endl;
		vector<Point2f> allimgpt[2];
		for (i = 0; i < nimages; i++)
			std::copy(imagePointsL[i].begin(), imagePointsL[i].end(), back_inserter(allimgpt[0]));
		for (i = 0; i < nimages; i++)
			std::copy(imagePointsR[i].begin(), imagePointsR[i].end(), back_inserter(allimgpt[1]));
		// 计算基础矩阵F
		F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
		Mat H1, H2;
		stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
		// 根据左右摄像头的内参数直接从基础矩阵F计算矫正变换矩阵
		R1 = cameraMatrixL.inv()*H1*cameraMatrixL;
		R2 = cameraMatrixR.inv()*H2*cameraMatrixR;
		P1 = cameraMatrixL;
		P2 = cameraMatrixR;
	}
	cout << "计算左右摄像头校正矩阵" << endl;
	Mat mxl, myl, mxr, myr;
	initUndistortRectifyMap(cameraMatrixL, distCoeffsL, R1, P1, imageSize, CV_16SC2, mxl, myl);
	initUndistortRectifyMap(cameraMatrixR, distCoeffsR, R2, P2, imageSize, CV_16SC2, mxr, myr);
	cout << "校正图像，计算视差图" << endl;
	RectifyComputeDisparity(point_cloud_filename, Q, mxl, myl, mxr, myr);
	return 0;
}
