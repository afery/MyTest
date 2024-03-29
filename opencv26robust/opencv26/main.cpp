//
//#include "stdafx.h"
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"//因为在属性中已经配置了opencv等目录，所以把其当成了本地目录一样
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
////#include <opencv2/nonfree/features2d.hpp>
//#include<opencv2/legacy/legacy.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <iostream>
//#include "opencv2/nonfree/nonfree.hpp"
//
//using namespace cv;
//using namespace std;
//
//void readme();
//
//int main(int argc,char* argv[])
//{
//
//
//	IplImage*iplImg = cvLoadImage("location (4).jpg",0);   //cvLoadImage( filename, 0 ); 强制转化读取图像为灰度图     cvLoadImage( filename, 1 ); 读取彩色图，默认读取方式。
//	Mat img_1(iplImg,true);
//	IplImage*iplImg2 = cvLoadImage("location (3).jpg",0);
//	Mat img_2(iplImg2,true);
//
//	// imshow("surf_Matches",img_1);//显示的标题为Matches
//	 //waitKey(10);
//
//	if(!img_1.data || !img_2.data)//如果数据为空
//	{
//		cout<<"opencv error"<<endl;
//		return -1;
//	}
//	cout<<"open right"<<endl;
//
//	//第一步，用SURF算子检测关键点
//	int minHessian=400;
//
//	SurfFeatureDetector detector(minHessian);
//	std::vector<KeyPoint> keypoints_1,keypoints_2;//构造2个专门由点组成的点向量用来存储特征点
//
//	detector.detect( img_1, keypoints_1 );//将img_1图像中检测到的特征点存储起来放在keypoints_1中
//
//	detector.detect( img_2, keypoints_2 );//同理
//
//
//	//在图像中画出特征点
//	Mat img_keypoints_1,img_keypoints_2;
//
//
//	drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//	drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//
//// 	imshow("surf_keypoints_1",img_keypoints_1);
//// 	imshow("surf_keypoints_2",img_keypoints_2);
//// 	waitKey(0);
//	//计算特征向量
//	SurfDescriptorExtractor extractor;//定义描述子对象
//
//	Mat descriptors_1,descriptors_2;//存放特征向量的矩阵
//
//	extractor.compute(img_1,keypoints_1,descriptors_1);
//	extractor.compute(img_2,keypoints_2,descriptors_2);
//
//	//用burte force进行匹配特征向量
//	BruteForceMatcher<L2<float>>matcher;//定义一个burte force matcher对象
//	vector<DMatch>matches;
//	matcher.match(descriptors_1,descriptors_2,matches);
//
//	//绘制匹配线段
//	Mat img_matches;
//	drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//将匹配出来的结果放入内存img_matches中
//
//	//显示匹配线段
//// 	imshow("surf_Matches",img_matches);//显示的标题为Matches
//// 	
//// 	waitKey(0);
//	imwrite("result.jpg",img_matches);
//	return 0;
//}