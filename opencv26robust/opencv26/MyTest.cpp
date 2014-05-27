//#pragma   comment   (lib, "vfw32.lib ")
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <ctime>
#include <string>
#include <fstream>
using namespace cv;
using namespace std;
//#define SIFT 
#define  scale 2

int main(){
	cv::initModule_nonfree();
	ofstream fout;
	fout.open("compare_flann.txt");

	char name[]="location ";
	string img_num=name;
	string img_name,img_name1;
	int num=15;
	int num1=16;
	char* cn=new char; 
	Mat image1,image2;
	Mat featureImage1,featureImage2;
	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints1a;	
	std::vector<cv::KeyPoint> keypoints2a;
	cv::SurfFeatureDetector surf(400.); // threshold 
	SiftFeatureDetector siftd(100);
	cv::SurfDescriptorExtractor surfDesc;
	SiftDescriptorExtractor siftDesc;
	std::vector< DMatch > good_matches;
	vector<vector<DMatch>> knnmatcher;
	// Extraction of the SURF descriptors
	cv::Mat descriptors1a;
	cv::Mat descriptors2a;

	BFMatcher matcher( cv::NORM_L2, false );
//	FlannBasedMatcher matcher;
//	BFMatcher matcher1( cv::NORM_L2, false );


// 	ratioTest(matches1);
// 	ratioTest(matches2);
// 
// 	symmetryTest(matches1, matches2, symMatches);
	// Match the two image descriptors
	std::vector<cv::DMatch> matchesa;
//	const float minRatio=1.f / 1.5f; 
	const float minRatio=0.6f; 
	const float minRatio_sift=0.8f; 
	int result_num;
	int total_pics=2;
	int begin_pics=9;
	cv::Mat imageMatchesa;
	//-----------------------------------------------------
	char picSave[]="p";
	static int picNum=1;
	for (num=begin_pics;num<begin_pics+total_pics-1;num++)
	{
		for (num1=num+1;num1<begin_pics+total_pics;num1++)
		{
			if (num==num1)
			{
				continue;
			}
			sprintf(cn,"(%d).jpg",num);
			img_name=img_num+cn;
			image1=imread(img_name);
#ifdef SIFT
			Mat image1_s(Size(image1.cols/scale,image1.rows/scale),CV_8UC3);
	
			cv::resize(image1,image1_s,image1_s.size());
#endif

			sprintf(cn,"(%d).jpg",num1);
			img_name1=img_num+cn;
			image2=imread(img_name1);
#ifdef SIFT
			Mat image2_s(Size(image2.cols/scale,image2.rows/scale),CV_8UC3);
			cv::resize(image2,image2_s,image2_s.size());
			featureImage1=image1_s;
			featureImage2=image2_s;
#else
			featureImage1=image1;
			featureImage2=image2;

#endif			//cv::OrbFeatureDetector Orb;
			// Detect the SURF features
#ifdef SIFT
			siftd.detect(featureImage1,keypoints1a);//sift
			siftd.detect(featureImage2,keypoints2a);//sift

			siftDesc.compute(featureImage1,keypoints1a,descriptors1a);//sift
			siftDesc.compute(featureImage2,keypoints2a,descriptors2a);//sift
#else
			// Construct the SURF feature detector object
 			surf.detect(featureImage1,keypoints1a);//surf
 			surf.detect(featureImage2,keypoints2a);//surf

			surfDesc.compute(featureImage1,keypoints1a,descriptors1a);           //surf
			surfDesc.compute(featureImage2,keypoints2a,descriptors2a);//surf
#endif


			//--------------------------------阈值剔除错匹配对------------------------
			good_matches.clear();  

			matcher.knnMatch(descriptors1a, descriptors2a, knnmatcher, 2);
			for (int i=0; i<knnmatcher.size(); i++)  
			{  
				const DMatch& bestMatch=knnmatcher[i][0];  
				const DMatch& betterMatch=knnmatcher[i][1];  

				float distanceRatio=bestMatch.distance/betterMatch.distance;  
#ifdef SIFT

				if (distanceRatio<minRatio_sift)  //sift or surf
				{  
					good_matches.push_back(bestMatch);  
				}  
#else

				if (distanceRatio<minRatio)  //sift or surf
				{  
					good_matches.push_back(bestMatch);  
				}  
#endif

			}  
			result_num=good_matches.size();
//			fout<<num<<" "<<num1<<" "<<result_num<<endl;
//			fout<<" "<<result_num<<" ";
 			string n=picSave;
 			char* pn=new char;
 			sprintf(pn,"%d.jpg",picNum++);
 			n+=pn;
		
			//------------------------------极线约束----------------------
			if (result_num<4)
			{
				fout<<" "<<result_num<<" ";
				continue;
			}
			int ptCount = (int)good_matches.size();
			Mat p1(ptCount, 2, CV_32F);
			Mat p2(ptCount, 2, CV_32F);
			Point2f pt;
			for (int i=0; i<ptCount; i++)
			{
				pt = keypoints1a[good_matches[i].queryIdx].pt;
				p1.at<float>(i, 0) = pt.x;
				p1.at<float>(i, 1) = pt.y;

				pt = keypoints2a[good_matches[i].trainIdx].pt;
				p2.at<float>(i, 0) = pt.x;
				p2.at<float>(i, 1) = pt.y;
			}

			// 用RANSAC方法计算F
			 Mat m_Fundamental;
			// 上面这个变量是基本矩阵
			vector<uchar> m_RANSACStatus;
			// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
			m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);

			// 计算野点个数
			int OutlinerCount = 0;
			for (int i=0; i<ptCount; i++)
			{
				if (m_RANSACStatus[i] == 0) // 状态为0表示野点
				{
					OutlinerCount++;
				}
			}

			// 计算内点
			vector<Point2f> m_LeftInlier;
			vector<Point2f> m_RightInlier;
			vector<DMatch> m_InlierMatches;
			// 上面三个变量用于保存内点和匹配关系
			int InlinerCount = ptCount - OutlinerCount;
			m_InlierMatches.resize(InlinerCount);
			m_LeftInlier.resize(InlinerCount);
			m_RightInlier.resize(InlinerCount);
			InlinerCount = 0;
			for (int i=0; i<ptCount; i++)
			{
				if (m_RANSACStatus[i] != 0)
				{
					m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
					m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
					m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
					m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
					m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
					m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
					InlinerCount++;
				}
			}
			
// 			if (m_InlierMatches.size()>3)
// 			{
// 				// 把内点转换为drawMatches可以使用的格式
// 				vector<KeyPoint> key1(InlinerCount);
// 				vector<KeyPoint> key2(InlinerCount);
// 				KeyPoint::convert(m_LeftInlier, key1);
// 				KeyPoint::convert(m_RightInlier, key2);
// 				drawMatches(image1, key1, image2, key2, m_InlierMatches, imageMatchesa);
// 				imwrite(n,imageMatchesa);
// 			}

			fout<<" "<<m_InlierMatches.size()<<" ";
			std::cout<<result_num<<" "<<m_InlierMatches.size()<<endl;
//--------------------------------------结果画出来~------------------------------------- 
  			 	cv::drawMatches(
  			 		featureImage1,keypoints1a, // 1st image and its keypoints
  			 		featureImage2,keypoints2a, // 2nd image and its keypoints
  			 		good_matches,            // the matches
  			 		imageMatchesa,      // the image produced
  			 		cv::Scalar(0,0,255)); // color of the lines
				imwrite("result_.jpg",imageMatchesa);

  		
// 			 	std::vector<Point2f> obj;
// 			 	std::vector<Point2f> scene;
// 			 	//double e8= (double)getTickCount();
// 			 	for( int i = 0; i < good_matches.size(); i++ )
// 			 	{
// 			 		//   -- Get the keypoints from the good matches
// 			 		obj.push_back( keypoints1a[ good_matches[i].queryIdx ].pt );
// 			 		scene.push_back( keypoints2a[ good_matches[i].trainIdx ].pt ); 
// 			 	}
// 				if (result_num<4)
// 				{
// 					cout<<endl;
// 					continue;
// 				}
// 			 	cv::Mat homography= cv::findHomography(obj, // corresponding 
// 			 		scene,
// 			 		CV_RANSAC  // RANSAC method 
// 			 		);         // max distance to reprojection point
// 			// 	double e9= (double)getTickCount();
// 			 	std::cout<<"H:"<<homography<<std::endl;
// 				std::vector<Point2f> obj_corners(4);
// 				obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
// 				obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
// 				std::vector<Point2f> scene_corners(4);
// 
// 				perspectiveTransform( obj_corners, scene_corners, homography);
// 
// 
// 				//-- Draw lines between the corners (the mapped object in the scene - image_2 )
// 				line( imageMatchesa, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 				line( imageMatchesa, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 				line( imageMatchesa, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 				line( imageMatchesa, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
			

		}
		
		fout<<endl;
	}
	// vector<KeyPoint> m_LeftKey;
	// vector<KeyPoint> m_RightKey;
	// vector<DMatch> m_Matches;
	// 以上三个变量已经被计算出来，分别是提取的关键点及其匹配，下面直接计算F

// 	// 分配空间
// 	int ptCount = (int)m_Matches.size();
// 	Mat p1(ptCount, 2, CV_32F);
// 	Mat p2(ptCount, 2, CV_32F);
// 
// 	// 把Keypoint转换为Mat
// 	Point2f pt;
// 	for (int i=0; i<ptCount; i++)
// 	{
// 		pt = m_LeftKey[m_Matches[i].queryIdx].pt;
// 		p1.at<float>(i, 0) = pt.x;
// 		p1.at<float>(i, 1) = pt.y;
// 
// 		pt = m_RightKey[m_Matches[i].trainIdx].pt;
// 		p2.at<float>(i, 0) = pt.x;
// 		p2.at<float>(i, 1) = pt.y;
// 	}
// 
// 	// 用RANSAC方法计算F
// 	// Mat m_Fundamental;
// 	// 上面这个变量是基本矩阵
// 	// vector<uchar> m_RANSACStatus;
// 	// 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
// 	m_Fundamental = findFundamentalMat(p1, p2, m_RANSACStatus, FM_RANSAC);
// 
// 	// 计算野点个数
// 	int OutlinerCount = 0;
// 	for (int i=0; i<ptCount; i++)
// 	{
// 		if (m_RANSACStatus[i] == 0) // 状态为0表示野点
// 		{
// 			OutlinerCount++;
// 		}
// 	}
// 
// 	// 计算内点
// 	// vector<Point2f> m_LeftInlier;
// 	// vector<Point2f> m_RightInlier;
// 	// vector<DMatch> m_InlierMatches;
// 	// 上面三个变量用于保存内点和匹配关系
// 	int InlinerCount = ptCount - OutlinerCount;
// 	m_InlierMatches.resize(InlinerCount);
// 	m_LeftInlier.resize(InlinerCount);
// 	m_RightInlier.resize(InlinerCount);
// 	InlinerCount = 0;
// 	for (int i=0; i<ptCount; i++)
// 	{
// 		if (m_RANSACStatus[i] != 0)
// 		{
// 			m_LeftInlier[InlinerCount].x = p1.at<float>(i, 0);
// 			m_LeftInlier[InlinerCount].y = p1.at<float>(i, 1);
// 			m_RightInlier[InlinerCount].x = p2.at<float>(i, 0);
// 			m_RightInlier[InlinerCount].y = p2.at<float>(i, 1);
// 			m_InlierMatches[InlinerCount].queryIdx = InlinerCount;
// 			m_InlierMatches[InlinerCount].trainIdx = InlinerCount;
// 			InlinerCount++;
// 		}
// 	}
// 
// 	// 把内点转换为drawMatches可以使用的格式
// 	vector<KeyPoint> key1(InlinerCount);
// 	vector<KeyPoint> key2(InlinerCount);
// 	KeyPoint::convert(m_LeftInlier, key1);
// 	KeyPoint::convert(m_RightInlier, key2);
// 
// 	// 显示计算F过后的内点匹配
// 	// Mat m_matLeftImage;
// 	// Mat m_matRightImage;
// 	// 以上两个变量保存的是左右两幅图像
// 	Mat OutImage;
// 	drawMatches(m_matLeftImage, key1, m_matRightImage, key2, m_InlierMatches, OutImage);
// 	cvNamedWindow( "Match features", 1);
// 	cvShowImage("Match features", &(IplImage(OutImage)));
// 	cvWaitKey( 0 );
// 	cvDestroyWindow( "Match features" );
	int a ;
	cout<<"The end."<<endl;
	cin>>a;

	



	// Draw the keypoints with scale and orientation information
	// cv::drawKeypoints(image1,      // original image
	// 				  keypoints1,               // vector of keypoints
	// 				  featureImage1,            // the resulting image
	// 				  cv::Scalar(255,255,255),   // color of the points
	// 				  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
	// imshow("1",featureImage1);
	// waitKey(0);
	// cv::drawKeypoints(image2,      // original image
	// 				  keypoints2,               // vector of keypoints
	// 				  featureImage2,            // the resulting image
	// 				  cv::Scalar(255,0,255),   // color of the points
	// 				  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
	//imshow("2",featureImage2);
	//waitKey(0);
	// Construction of the SURF descriptor extractor 
	

	


	
	// Construction of the matcher 
	//cv::BFMatcher<cv::L2<float>> matcher;

//	matcher.match(descriptors1a,descriptors2a, matchesa);
	

// 	double q5= (double)getTickCount();
// 	std::nth_element(matchesa.begin(),    // initial position
// 		matchesa.begin()+20, // position of the sorted element
// 		matchesa.end());     // end position
// 	// remove all elements after the 25th
// 	matchesa.erase(matchesa.begin()+21, matchesa.end()); 

// 	
// 	cv::Mat imageMatchesa;
// 
// 	cv::drawMatches(
// 		image1,keypoints1a, // 1st image and its keypoints
// 		image2,keypoints2a, // 2nd image and its keypoints
// 		good_matches,            // the matches
// 		imageMatchesa,      // the image produced
// 		cv::Scalar(0,0,255)); // color of the lines

	//imshow("a",imageMatchesa);
	//waitKey(0);
	/*std::nth_element(matchesb.begin(),    // initial position
	matchesb.begin()+10, // position of the sorted element
	matchesb.end());     // end position
	// remove all elements after the 25th
	matchesb.erase(matchesb.begin()+11, matchesb.end()); */


	//imshow("2",imageMatchesb);
	//waitKey(0);
	// Find the homography between image 1 and image 2
	//std::vector<uchar> inliers(points1.size(),0);
	//	double e5= (double)getTickCount();
// 	std::vector<Point2f> obj;
// 	std::vector<Point2f> scene;
// 	double e8= (double)getTickCount();
// 	for( int i = 0; i < good_matches.size(); i++ )
// 	{
// 		//   -- Get the keypoints from the good matches
// 		obj.push_back( keypoints1a[ good_matches[i].queryIdx ].pt );
// 		scene.push_back( keypoints2a[ good_matches[i].trainIdx ].pt ); 
// 	}

// 	cv::Mat homography= cv::findHomography(obj, // corresponding 
// 		scene,
// 		CV_RANSAC  // RANSAC method 
// 		);         // max distance to reprojection point
// 	double e9= (double)getTickCount();
// 	std::cout<<"H:"<<(e9-e8)*1000/(CLOCKS_PER_SEC)<<std::endl;
// 	std::vector<Point2f> obj_corners(4);
// 	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
// 	obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
// 	std::vector<Point2f> scene_corners(4);
// 
// 	perspectiveTransform( obj_corners, scene_corners, homography);
// 
// 
// 	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
// 	line( imageMatchesa, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 	line( imageMatchesa, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 	line( imageMatchesa, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 	line( imageMatchesa, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
// 	imwrite("a03.jpg",imageMatchesa);
// 
// 	while (1);


}

// int ratioTest( VVecMatch &matches )
// {
// 	float ratio = 0.8f;
// 	int removed=0;
// 	// for all matches
// 	for (std::vector<std::vector>::iterator
// 		matchIterator= matches.begin();
// 		matchIterator!= matches.end(); ++matchIterator)
// 	{
// 		// if 2 NN has been identified
// 		if (matchIterator->size() > 1)
// 		{
// 			// check distance ratio
// 			if ((*matchIterator)[0].distance/
// 				(*matchIterator)[1].distance > ratio) {
// 					matchIterator->clear(); // remove match
// 					removed++;
// 			}
// 		} else { // does not have 2 neighbours
// 			matchIterator->clear(); // remove match
// 			removed++;
// 		}
// 	}
// 	return removed;
// }


// void symmetryTest( const VVecMatch matches1,
// 	const VVecMatch matches2,
// 	VecMatch& symMatches )
// {
// 	// for all matches image 1 -> image 2
// 	for (std::vector<std::vector>::const_iterator matchIterator1= matches1.begin();
// 		matchIterator1!= matches1.end(); ++matchIterator1)
// 	{
// 		// ignore deleted matches
// 		if (matchIterator1->size() < 2)
// 			continue;
// 		// for all matches image 2 -> image 1
// 		for (std::vector<std::vector>::const_iterator matchIterator2= matches2.begin();
// 			matchIterator2!= matches2.end();
// 			++matchIterator2)
// 		{
// 			// ignore deleted matches
// 			if (matchIterator2->size() < 2)
// 				continue;
// 			// Match symmetry test
// 			if ((*matchIterator1)[0].queryIdx ==
// 				(*matchIterator2)[0].trainIdx &&
// 				(*matchIterator2)[0].queryIdx ==
// 				(*matchIterator1)[0].trainIdx)
// 			{
// 				// add symmetrical match
// 				symMatches.push_back(
// 					cv::DMatch((*matchIterator1)[0].queryIdx,
// 					(*matchIterator1)[0].trainIdx,
// 					(*matchIterator1)[0].distance));
// 				break; // next match in image 1 -> image 2
// 			}
// 		}
// 	}
// }



