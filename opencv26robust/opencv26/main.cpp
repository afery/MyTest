//
//#include "stdafx.h"
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"//��Ϊ���������Ѿ�������opencv��Ŀ¼�����԰��䵱���˱���Ŀ¼һ��
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
//	IplImage*iplImg = cvLoadImage("location (4).jpg",0);   //cvLoadImage( filename, 0 ); ǿ��ת����ȡͼ��Ϊ�Ҷ�ͼ     cvLoadImage( filename, 1 ); ��ȡ��ɫͼ��Ĭ�϶�ȡ��ʽ��
//	Mat img_1(iplImg,true);
//	IplImage*iplImg2 = cvLoadImage("location (3).jpg",0);
//	Mat img_2(iplImg2,true);
//
//	// imshow("surf_Matches",img_1);//��ʾ�ı���ΪMatches
//	 //waitKey(10);
//
//	if(!img_1.data || !img_2.data)//�������Ϊ��
//	{
//		cout<<"opencv error"<<endl;
//		return -1;
//	}
//	cout<<"open right"<<endl;
//
//	//��һ������SURF���Ӽ��ؼ���
//	int minHessian=400;
//
//	SurfFeatureDetector detector(minHessian);
//	std::vector<KeyPoint> keypoints_1,keypoints_2;//����2��ר���ɵ���ɵĵ����������洢������
//
//	detector.detect( img_1, keypoints_1 );//��img_1ͼ���м�⵽��������洢��������keypoints_1��
//
//	detector.detect( img_2, keypoints_2 );//ͬ��
//
//
//	//��ͼ���л���������
//	Mat img_keypoints_1,img_keypoints_2;
//
//
//	drawKeypoints(img_1,keypoints_1,img_keypoints_1,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//	drawKeypoints(img_2,keypoints_2,img_keypoints_2,Scalar::all(-1),DrawMatchesFlags::DEFAULT);
//
//// 	imshow("surf_keypoints_1",img_keypoints_1);
//// 	imshow("surf_keypoints_2",img_keypoints_2);
//// 	waitKey(0);
//	//������������
//	SurfDescriptorExtractor extractor;//���������Ӷ���
//
//	Mat descriptors_1,descriptors_2;//������������ľ���
//
//	extractor.compute(img_1,keypoints_1,descriptors_1);
//	extractor.compute(img_2,keypoints_2,descriptors_2);
//
//	//��burte force����ƥ����������
//	BruteForceMatcher<L2<float>>matcher;//����һ��burte force matcher����
//	vector<DMatch>matches;
//	matcher.match(descriptors_1,descriptors_2,matches);
//
//	//����ƥ���߶�
//	Mat img_matches;
//	drawMatches(img_1,keypoints_1,img_2,keypoints_2,matches,img_matches);//��ƥ������Ľ�������ڴ�img_matches��
//
//	//��ʾƥ���߶�
//// 	imshow("surf_Matches",img_matches);//��ʾ�ı���ΪMatches
//// 	
//// 	waitKey(0);
//	imwrite("result.jpg",img_matches);
//	return 0;
//}