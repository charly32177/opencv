#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"
#include <time.h>
#include <stdio.h>
using namespace cv;
bool comparearea (vector<Point> , vector<Point>  );
int main(int argc, char *argv[])
{
  //  double timestamp = 1000.0*clock()/CLOCKS_PER_SEC;
  VideoCapture cap(0); // open the default camera
//  VideoCapture cap1(1);
  if(!cap.isOpened()){  // check if we succeeded
	return -1;
	std::cout << "camera1 not work" << std::endl;
  }	
//  if(!cap1.isOpened()){
//	return -1;
//	std::cout << "camera2 not work" << std::endl;
 // }
  //  vector<Rect> seg_bounds;
  int takepicture = 0;
  int count = 0;
  int history = 500;
  int during =0;
  double dist2Threshold = 400;
  double varThreshold = 8;
  bool detectShadows = true ;
  Ptr<BackgroundSubtractor> pMOG2;

  pMOG2 = new BackgroundSubtractorMOG2(history, varThreshold, detectShadows);
  //	pMOG2 = createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
  //	pKNN =  createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);

  bool update_bg_model = true;
  Mat frame , frame2;
  Mat fgMOG2MaskImg, fgMOG2Img;
  Mat fgKNNMaskImg, fgKNNImg,tdila ,dila, withpoint;
  while(1)
  {
	cap >> frame;
//	cap1>>frame2;
	if( frame.empty() ){
	  break;
	}
//	if(frame2.empty()){
//	  break;
//	}

	if( fgMOG2Img.empty() )
	  fgMOG2Img.create(frame.size(), frame.type());
	//		if( fgKNNImg.empty() )
	//			fgKNNImg.create(frame.size(), frame.type());


	//update the model
	pMOG2->operator()(frame, fgMOG2MaskImg,update_bg_model ? -1 : 0);
	//		pKNN->operator()(frame, fgKNNMaskImg, update_bg_model ? -1 : 0);

	fgMOG2Img = Scalar::all(0);
	//		fgKNNImg = cv::Scalar::all(0);

	frame.copyTo(fgMOG2Img, fgMOG2MaskImg);
	//		frame.copyTo(fgKNNImg, fgKNNMaskImg);

	Mat bgMOG2Img, bgKNNImg;
	pMOG2->getBackgroundImage(bgMOG2Img);
	//		pKNN->getBackgroundImage(bgKNNImg);
	medianBlur(fgMOG2MaskImg,fgMOG2MaskImg,7);
//	boxFilter(fgMOG2MaskImg,fgMOG2MaskImg,-1,Size(3,3));
	threshold(fgMOG2MaskImg,tdila,64,255,THRESH_BINARY);

	Mat element = getStructuringElement( MORPH_RECT,
		Size( 2*3 + 1, 2*3+1 )
		);

	// Apply the erosion operation
		dilate( tdila, dila, element );
	//	for (int i =0 ; i< 10; i++)
	//	{
	//		erode(dila,dila,element );
	//		dilate( dila, dila, element );
		erode(dila,dila,element );
	// 		dilate( dila, dila, element );
	//	}
	//  	imshow( "Erosion Demo", erosion_dst );

	//	SimpleBlobDetector detector;
	//	SimpleBlobDetector::Params params;
	//	params.filterByColor = true;
	//	params.blobColor=255;
	//	SimpleBlobDetector detector(params);
	//	std::vector<KeyPoint> keypoints;
	//	detector.detect(dila,keypoints);
	//	drawKeypoints(tdila,keypoints,withpoint,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//	  Mat findBiggestBlob(cv::Mat & matImage){
	  int largest_area=0;
	  int largest_contour_index=0;
      int slargest_area=0;
      int slargest_contour_index=0;

	  vector< vector<Point> > contours; // Vector for storing contour
//      vector< vector<Point> > tmp;
//		tmp[0][0] = 0.0;
	  vector<Vec4i> hierarchy;
	  Rect bounding_rect;
      Rect bounding_rect1;
      Rect bounding_rect2;
      Rect bounding_rect3;
      Rect bounding_rect4;
      Rect bounding_rect5;

	  findContours( dila, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ); // Find the contours in the image
//      findContours( dila, tmp, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
//	  for(int i = 0 ; i < contours.size() ; i++){
//	  	for( int pass = i; pass > 0; pass-- ) {
// 			double a=contourArea( contours[pass],false); 
//			double a_1=contourArea( contours[pass-1],false);
//	      	if(a>a_1){
//				tmp[0][0] = contours[pass];
//				contours[pass] = contours[pass-1];
//				contours[pass-1]=tmp[0][0];
//			}
//		}
//	  } 
//	  std::sort(contours.begin(),contours.end(),comparearea )	;	
	  for( int i = 0; i< contours.size(); i++ ) {// iterate through each contour. 
		double a=contourArea( contours[i],false);  //  Find the area of contour
		if(a>largest_area){
		  largest_area=a;
		  largest_contour_index=i;                //Store the index of largest contour
		   bounding_rect=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
      //     bounding_rect1=boundingRect(contours[1]);
//           bounding_rect2=boundingRect(contours[2]);
//           bounding_rect3=boundingRect(contours[3]);
//           bounding_rect4=boundingRect(contours[4]);
//           bounding_rect5=boundingRect(contours[5]);

//		}
		}
	  }
//for( int i = 0; i< contours.size(); i++ ) {// iterate through each contour. 
 //        double a=contourArea( contours[i],false);  //  Find the area of contour
  //       if(a>slargest_area && a<largest_area){
   //        slargest_area=a;
    //       slargest_contour_index=i;                //Store the index of largest contour
     //      bounding_rect1=boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
    //     }
     //  }

//		std::cout<<"i= "<< largest_contour_index <<"size = "<< contours.size()  <<std::endl;
//	  drawContours( tdila, contours, largest_contour_index, Scalar(255), CV_FILLED, 8, hierarchy ); // Draw the largest contour using previously stored index.

	  int posx = std::max(bounding_rect.x -10 , 0);
      int posy = std::max(bounding_rect.y -10 , 0);
	  string boxtext = format("Objection alpha");
      //rectangle(frame,bounding_rect1 ,Scalar(0,0,255), 1);
   //   rectangle(frame,bounding_rect2 ,Scalar(0,0,255), 1);
  //    rectangle(frame,bounding_rect3 ,Scalar(0,0,255), 1);
 //     rectangle(frame,bounding_rect4 ,Scalar(0,0,255), 1);
//      rectangle(frame,bounding_rect5 ,Scalar(0,0,255), 1);
//	  return matImage;
//	}
	time_t t;
	t=time(NULL);
	char TIMEW[64];
	sprintf (TIMEW,"%s\n",ctime(&t));
	string nowtime = format(TIMEW);
//	std::cout<< TIMEW << std::endl;
	putText(frame,nowtime,Point(10,15),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,0,255),2.0);
	double sum_diff = 0;
	for (int i = 0; i < tdila.cols; ++i) {
    	for (int j = 0; j < tdila.rows; ++j)
        	sum_diff += abs(tdila.at<uchar> (j ,i));
    }
	std::cout<< sum_diff/(float)(tdila.cols * tdila.rows) << std::endl;
	char DIFF[64];
	sprintf (DIFF,"%f",sum_diff/(float)(tdila.cols * tdila.rows));
	string sum_diff2 = format(DIFF);
	putText(frame,sum_diff2,Point(10,35),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,0,255),2.0);
	if(sum_diff/(float)(tdila.cols * tdila.rows) > 2)
	{
		during ++ ;
		takepicture = 1;
		rectangle(frame,bounding_rect ,Scalar(0,0,255), 1);
		putText(frame,boxtext,Point(posx,posy),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
//		putText(frame,boxtext,Point(posx,posy-10),FONT_HERSHEY_PLAIN,1.0,CV_RGB(0,255,0),2.0);
	}
	if(takepicture = 1 && during > 24)
	{
		count++;
		char pic[64];
	    sprintf (pic,"./picture2/%d_%s.jpg",count,ctime(&t));
		imwrite(pic,frame);
				
	}
	if (count > 1440)
	{
		count =0;
		takepicture = 0;
		during =0;
	}
	imshow("frame", frame);
//	imshow("frame2", frame2);
	imshow("MOG2 Foreground Mask",tdila);
	//		cv::imshow("KNN Foreground Mask", fgKNNMaskImg);
//	imshow("MOG2 Foreground Image", fgMOG2Img);
	//		cv::imshow("KNN Foreground Image", fgKNNImg);

	//		if(!bgMOG2Img.empty())
	//			imshow("MOG2 Mean Background Image", bgMOG2Img );
	//		if(!fgKNNImg.empty())
	//			cv::imshow("KNN Mean Background Image", bgKNNImg );

	//		waitKey(30);
	if(waitKey(30) >= 0) break;
  }

  return 0;
}
//       bool comparearea (vector<Point>  contours1 , vector<Point> contours2 )
//       {
//         double i = fabs (contourArea( contours1));
//         double j = fabs (contourArea( contours2));
//        return (i<j);
 
//       }

