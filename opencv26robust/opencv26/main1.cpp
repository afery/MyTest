////#pragma   comment   (lib, "vfw32.lib ")
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
////#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <ctime>
//using namespace cv;
//
//
//
//int main(){
//
//	Mat image1=imread("Lena.jpg");
//	Mat image2=imread("Lenaa.jpg");
//	Mat featureImage1(image1);
//	Mat featureImage2(image2);
//	// vector of keypoints
//	std::vector<cv::KeyPoint> keypoints1a;
//	std::vector<cv::KeyPoint> keypoints1b;
//	std::vector<cv::KeyPoint> keypoints2a;
//	std::vector<cv::KeyPoint> keypoints2b;
//	// Construct the SURF feature detector object
//	cv::SurfFeatureDetector surf(2500.); // threshold 
//	cv::OrbFeatureDetector Orb;
//	// Detect the SURF features
//	double s1 = (double)getTickCount();
//	surf.detect(image1,keypoints1a);
//	double s2 = (double)getTickCount();;
//	Orb.detect(image1,keypoints1b);
//	double s3 = (double)getTickCount();
//	surf.detect(image2,keypoints2a);
//	double s4 = (double)getTickCount();
//	Orb.detect(image2,keypoints2b);
//	double s5 = (double)getTickCount();
//	std::cout<<"surfdetect1:"<<(s2-s1)/getTickFrequency()<<" "<<"surfdetect2:"<<(s4-s3)/getTickFrequency()<<std::endl;
//	std::cout<<"orbdetect1:"<<(s3-s2)/getTickFrequency()<<" "<<"orbdetect2:"<<(s5-s4)/getTickFrequency()<<std::endl;
//	// Draw the keypoints with scale and orientation information
//	// cv::drawKeypoints(image1,      // original image
//	// 				  keypoints1,               // vector of keypoints
//	// 				  featureImage1,            // the resulting image
//	// 				  cv::Scalar(255,255,255),   // color of the points
//	// 				  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
//	// imshow("1",featureImage1);
//	// waitKey(0);
//	// cv::drawKeypoints(image2,      // original image
//	// 				  keypoints2,               // vector of keypoints
//	// 				  featureImage2,            // the resulting image
//	// 				  cv::Scalar(255,0,255),   // color of the points
//	// 				  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag
//	//imshow("2",featureImage2);
//	//waitKey(0);
//	// Construction of the SURF descriptor extractor 
//	cv::SurfDescriptorExtractor surfDesc;
//	cv::OrbDescriptorExtractor orbDesc;
//	// Extraction of the SURF descriptors
//	cv::Mat descriptors1a;
//	cv::Mat descriptors1b;
//	cv::Mat descriptors2a;
//	cv::Mat descriptors2b;
//	double e1 = (double)getTickCount();
//	surfDesc.compute(image1,keypoints1a,descriptors1a);
//	double e2 = (double)getTickCount();
//	orbDesc.compute(image1,keypoints1b,descriptors1b);
//	double e3 = (double)getTickCount();
//	surfDesc.compute(image2,keypoints2a,descriptors2a);
//	double e4= (double)getTickCount();
//	orbDesc.compute(image2,keypoints2b,descriptors2b);
//	double e5= (double)getTickCount();
//	std::cout<<"surfdes1:"<<(e2-e1)/getTickFrequency()<<" "<<"surfdes2:"<<(e4-e3)/getTickFrequency()<<std::endl;
//	std::cout<<"surfdes1:"<<(e3-e2)/getTickFrequency()<<" "<<"surfdes2:"<<(e5-e4)/getTickFrequency()<<std::endl;
//	// Construction of the matcher 
//	//cv::BFMatcher<cv::L2<float>> matcher;
//	BFMatcher matcher( cv::NORM_L2, false );
//	//FlannBasedMatcher matcher;
//	// Match the two image descriptors
//	std::vector<cv::DMatch> matchesa;
//	std::vector<cv::DMatch> matchesb;
//	double q6= (double)getTickCount();
//	matcher.match(descriptors1a,descriptors2a, matchesa);
//	matcher.match(descriptors1b,descriptors2b, matchesb);
//	double q5= (double)getTickCount();
//	std::nth_element(matchesa.begin(),    // initial position
//		matchesa.begin()+20, // position of the sorted element
//		matchesa.end());     // end position
//	// remove all elements after the 25th
//	matchesa.erase(matchesa.begin()+21, matchesa.end()); 
//	double q7= (double)getTickCount();
//	std::cout<<"1:"<<(q5-q6)*1000/(CLOCKS_PER_SEC)<<" "<<"2:"<<(q7-q5)*1000/(CLOCKS_PER_SEC)<<std::endl;
//	cv::Mat imageMatchesa;
//	cv::Mat imageMatchesb;
//	cv::drawMatches(
//		image1,keypoints1a, // 1st image and its keypoints
//		image2,keypoints2a, // 2nd image and its keypoints
//		matchesa,            // the matches
//		imageMatchesa,      // the image produced
//		cv::Scalar(0,255,0)); // color of the lines
//
//	//imshow("a",imageMatchesa);
//	//waitKey(0);
//	/*std::nth_element(matchesb.begin(),    // initial position
//	matchesb.begin()+10, // position of the sorted element
//	matchesb.end());     // end position
//	// remove all elements after the 25th
//	matchesb.erase(matchesb.begin()+11, matchesb.end()); */
//	cv::drawMatches(
//		image1,keypoints1b, // 1st image and its keypoints
//		image2,keypoints2b, // 2nd image and its keypoints
//		matchesb,            // the matches
//		imageMatchesb,      // the image produced
//		cv::Scalar(0,255,0)); // color of the lines
//
//	//imshow("2",imageMatchesb);
//	//waitKey(0);
//	// Find the homography between image 1 and image 2
//	//std::vector<uchar> inliers(points1.size(),0);
//	//	double e5= (double)getTickCount();
//	std::vector<Point2f> obj;
//	std::vector<Point2f> scene;
//	double e8= (double)getTickCount();
//	for( int i = 0; i < matchesa.size(); i++ )
//	{
//		//   -- Get the keypoints from the good matches
//		obj.push_back( keypoints1a[ matchesa[i].queryIdx ].pt );
//		scene.push_back( keypoints2a[ matchesa[i].trainIdx ].pt ); 
//	}
//
//	cv::Mat homography= cv::findHomography(obj, // corresponding 
//		scene,
//		CV_RANSAC  // RANSAC method 
//		);         // max distance to reprojection point
//	double e9= (double)getTickCount();
//	std::cout<<"H:"<<(e9-e8)*1000/(CLOCKS_PER_SEC)<<std::endl;
//	std::vector<Point2f> obj_corners(4);
//	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
//	obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
//	std::vector<Point2f> scene_corners(4);
//
//	perspectiveTransform( obj_corners, scene_corners, homography);
//
//
//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//	line( imageMatchesa, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
//	line( imageMatchesa, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
//	line( imageMatchesa, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
//	line( imageMatchesa, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar(255, 0, 0), 4 );
//	imwrite("a03.jpg",imageMatchesa);
//	obj.clear();
//	scene.clear();
//
//	for( int i = 0; i < matchesb.size(); i++ )
//	{
//		//   -- Get the keypoints from the good matches
//		obj.push_back( keypoints1b[ matchesb[i].queryIdx ].pt );
//		scene.push_back( keypoints2b[ matchesb[i].trainIdx ].pt ); 
//	}
//	FILE *oa;
//	oa=fopen("oa.txt","w");
//	int Num=obj.size();
//	std::cout<<Num<<std::endl;
//	for(int i=0;i<Num;i++)
//		fprintf(oa,"%f,%f,",obj[i].x,obj[i].y);
//	fclose(oa);
//	FILE *ob;
//	ob=fopen("ob.txt","w");
//	int Num1=scene.size();
//	std::cout<<Num1<<std::endl;
//	for(int i=0;i<Num1;i++)
//		fprintf(ob,"%f,%f,",scene[i].x,scene[i].y);
//	fclose(ob);
//	clock_t tt1=clock();
//	homography= cv::findHomography(obj, // corresponding 
//		scene,
//		CV_RANSAC  // RANSAC method 
//		); 
//	clock_t tt2=clock();
//
//
//	//	std::vector<Point2f> obj_corners(4);
//	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
//	obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
//	//	std::vector<Point2f> scene_corners(4);
//
//	perspectiveTransform( obj_corners, scene_corners, homography);
//
//	std::cout<<"Pers:"<<(double)(tt2-tt1)*1000/CLOCKS_PER_SEC<<std::endl;
//
//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//	line( imageMatchesb, scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(0, 0, 255), 4 );
//	line( imageMatchesb, scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar(0, 0, 255), 4 );
//	line( imageMatchesb, scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar(0, 0, 255), 4 );
//	line( imageMatchesb, scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar(0, 0, 255), 4 );
//	imwrite("a04.jpg",imageMatchesb);
//	//while (1);
//
//
//}
