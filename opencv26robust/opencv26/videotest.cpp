//**
// * @file SURF_Homography
// * @brief SURF detector + descriptor + FLANN Matcher + FindHomography
// * @author A. Huaman
// */
////#pragma   comment   (lib, "vfw32.lib ")
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"
////#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include <opencv2/nonfree/features2d.hpp>
//#include <ctime>
//#include <opencv2/imgproc/imgproc.hpp>
//using namespace cv;
//using namespace std;
//
//void readme();
//
///**
// * @function main
// * @brief Main function	
// */
//int main( int argc, char** argv )
//{
////   if( argc != 3 )
////   { readme(); return -1; }
//	VideoCapture cap(4);
//	if(!cap.isOpened())
//	return -1;
//
//	
//  Mat img_object = imread("cup2.jpg" , CV_LOAD_IMAGE_GRAYSCALE );
//  Mat img_scene;  Mat img_matches;
////  Mat img_scene = imread( "cup3.jpg", CV_LOAD_IMAGE_GRAYSCALE );
//// imshow("main",img_object);
//// imshow("vice",img_scene);
// /* Mat img_object = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
//  Mat img_scene = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );*/
//	int minHessian = 400;
//	SiftFeatureDetector detector;
// //  ORBFeatureDetector detector(  );
//	std::vector<KeyPoint> keypoints_object, keypoints_scene;
//	
//	clock_t start0=clock();
//   detector.detect( img_object, keypoints_object );
//   clock_t end0=clock();
//   std::cout<<(float)(end0-start0)/CLOCKS_PER_SEC <<std::endl;
//   std::cout<< "Total: " << keypoints_object.size()  << std::endl; 
//    SiftDescriptorExtractor extractor;
//    //ORBDescriptorExtractor extractor;
//	 Mat descriptors_object, descriptors_scene;
//	 clock_t start1=clock();
//	 extractor.compute( img_object, keypoints_object, descriptors_object );
//	 clock_t end1=clock();
//	  std::cout<<(float)(end1-start1)/CLOCKS_PER_SEC <<std::endl;
//	 BFMatcher matcher1( cv::NORM_L2, false );
//	namedWindow("Good Matches & Object detection",1);
//  	for ( ; ;)
//	{
//        clock_t start3 = clock();
//		std::vector< DMatch > matches1;
//		Mat frame;
//		cap >> frame; // get a new frame from camera
//		cvtColor(frame, img_scene, CV_BGR2GRAY);
//		imshow("ha",img_scene);
//		waitKey();
//		if(  !img_scene.data )
//		continue;
//		detector.detect( img_scene, keypoints_scene );
//		std::cout<< "Total: " << keypoints_scene.size()  << std::endl;
//		if (keypoints_scene.size() ==0)
//		{
//			continue;
//		}
//		extractor.compute( img_scene, keypoints_scene, descriptors_scene );
//		matcher1.match( descriptors_object, descriptors_scene, matches1 );
//
//		 //std::cout<< "BF took: " << float(end - start) / CLOCKS_PER_SEC  << " seconds" << std::endl;
//		std::cout<< "matches: " << matches1.size()<< std::endl;
//		 double max_dist = 0; 
//		 double min_dist = 100;
//		 for( int i = 0; i < matches1.size(); i++ )	
//		 { 
//            //cout<<"test"<<i<<endl;
//            //cout<<"rows"<<descriptors_object.rows<<endl;
//            double dist = matches1[i].distance;
//            if( dist < min_dist ) min_dist = dist;
//            if( dist > max_dist ) max_dist = dist;
//		 }
//
//		 printf("-- Max dist : %f \n", max_dist );
//		 printf("-- Min dist : %f \n", min_dist );
//		 std::vector< DMatch > good_matches;
//		 
//		 if(matches1.size()<=0)
//		    continue;
//
//		 for( int i = 0; i < matches1.size(); i++ )
//		 { 
//			 if( matches1[i].distance < 2*min_dist )
//				good_matches.push_back( matches1[i]); 
//		 }  
//		 drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, 
//			 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
//			 vector<char>(), DrawMatchesFlags::DEFAULT ); 
//		 imshow("1",img_object);
//		 waitKey();
//		 std::vector<Point2f> obj;
//		 std::vector<Point2f> scene;
//
//		 for( int i = 0; i < good_matches.size(); i++ )
//		 {
//			 //-- Get the keypoints from the good matches
//			 obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
//			 scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt ); 
//		 }
//
//		// clock_t start3 = clock();
//		 Mat H = findHomography( obj, scene, CV_RANSAC );
//		 //clock_t end3 = clock();
//		// std::cout<< "Homo took: " << float(end3 - start3) / CLOCKS_PER_SEC  << " seconds" << std::endl;
//		// std::cout<< "Good: " <<good_matches.size() << std::endl;
//		 //-- Get the corners from the image_1 ( the object to be "detected" )
//		 std::vector<Point2f> obj_corners(4);
//		 obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
//		 obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
//		 std::vector<Point2f> scene_corners(4);
//
//		 perspectiveTransform( obj_corners, scene_corners, H);
//
//
//		 //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//		 line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//		 line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//		 line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//		 line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//
//		 //-- Show detected matches
//		 imshow( "Good Matches & Object detection", img_matches );
//		int c = cvWaitKey(10); //cvWaitKey(30);
//		if ((char)c==' ')
//		{
//			break;
//		}
//	   clock_t end3 = clock();
//	   cout<< " took: " <<  CLOCKS_PER_SEC/float(end3 - start3)   << " frame/s" << std::endl;	
//
//}   
//
// // -- Step 1: Detect the keypoints using SURF Detector
//
//
////   
////  
////  
////   
////   
////    clock_t start0 = clock();
////   
//// clock_t end0 = clock();
////   
////  std::cout<< "Detect took: " << float(end0 - start0) / CLOCKS_PER_SEC  << " seconds" << std::endl;
////   //-- Step 2: Calculate descriptors (feature vectors)
////  
//// 
////  
////   clock_t start1 = clock();
////   
////  clock_t end1 = clock();
////  std::cout<< "extract took: " << float(end1 - start1) / CLOCKS_PER_SEC  << " seconds" << std::endl;
//// 
////  clock_t start2 = clock();
//// 
//// 
////   clock_t end2 = clock();
////   std::cout<< "extract took: " << float(end2 - start2) / CLOCKS_PER_SEC  << " seconds" << std::endl;
//// 
////   //-- Step 3: Matching descriptor vectors using FLANN matcher
////    FlannBasedMatcher matcher0;
////    std::vector< DMatch > matches0;
////    clock_t start4 = clock();
////    matcher0.match( descriptors_object, descriptors_scene, matches0 );
////    clock_t end4 = clock();
////    std::cout<< "matches: " << matches0.size()<< std::endl;
////   std::cout<< "Flann took: " << float(end4 - start4) / CLOCKS_PER_SEC  << " seconds" << std::endl;
////  
////   clock_t start = clock();
////  
////   
////  
//// 
////   //-- Quick calculation of max and min distances between keypoints
//// 
////   
////   //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//// 
//// 
////   drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, 
////                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), 
////                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS ); 
//// 
//// 
////   //-- Localize the object from img_1 in img_2 
////   
////   waitKey(0);
//
////  return 0;
////}
//
///**
// * @function readme
// */
////void readme()
////{ std::cout << " Usage: ./SURF_Homography <img1> <img2>" << std::endl; }
///**
// * @file SURF_detector
// * @brief SURF keypoint detection + keypoint drawing with OpenCV functions
// * @author A. Huaman
// */
//
////#include <stdio.h>
////#include <iostream>
////#include "opencv2/core/core.hpp"
//////#include "opencv2/features2d/features2d.hpp"
////#include "opencv2/highgui/highgui.hpp"
////#include <opencv2/nonfree/features2d.hpp>
////using namespace cv;
////
////void readme();
//
///**
// * @function main
// * @brief Main function
// */
////int main( int argc, char** argv )
////{
////  if( argc != 3 )
////  { readme(); return -1; }
////
////  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
////  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
////  
////  if( !img_1.data || !img_2.data )
////  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
////
////  //-- Step 1: Detect the keypoints using SURF Detector
////  int minHessian = 400;
////
////  SurfFeatureDetector detector( minHessian );
////
////  std::vector<KeyPoint> keypoints_1, keypoints_2;
////
////  detector.detect( img_1, keypoints_1 );
////  detector.detect( img_2, keypoints_2 );
////
////  //-- Draw keypoints
////  Mat img_keypoints_1; Mat img_keypoints_2;
////
////  drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT ); 
////  drawKeypoints( img_2, keypoints_2, img_keypoints_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT ); 
////
////  //-- Show detected (drawn) keypoints
////  imshow("Keypoints 1", img_keypoints_1 );
////  imshow("Keypoints 2", img_keypoints_2 );
////
////  waitKey(0);
////
////  return 0;
////}
////
/////**
//// * @function readme
//// */
////void readme()
////{ std::cout << " Usage: ./SURF_detector <img1> <img2>" << std::endl; }
