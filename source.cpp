#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <opencv\cv.h>
#include <opencv\highgui.h>
#include <opencv2\core\core.hpp>
#include <opencv2\core\mat.hpp>
//#include <opencv2/core/core.hpp>        
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <math.h>
#include <fstream>
#include <string.h>
#include <time.h>
#include <cstdlib>
#include "lib_data.h"
#include "lib_histogram.h"
#include "lib_cut_object_from_background.h"
using namespace cv;
using namespace std;

Mat src_L,src_R;
Mat src_gray_R,src_gray_L;
Mat dst_L,dst_R;
Mat source_image_L, source_image_R;

void Hold_L(Mat& im, int type, int value, Mat& result)
{
	src_L = im;
  /// Convert the image to Gray
  cvtColor( src_L, src_gray_L, CV_RGB2GRAY );
  
  /// Create to choose type of Threshold
  threshold_type = type;
  threshold_value = value;
  /// Call the function to initialize
  threshold( src_gray_L, dst_L, threshold_value, max_BINARY_value,threshold_type );
  /// Wait until user finishes program
result = dst_L;
}
void Hold_R(Mat& im, int type, int value, Mat& result)
{
	src_R = im;
  /// Convert the image to Gray
  cvtColor( src_R, src_gray_R, CV_RGB2GRAY );
 
  /// Create to choose type of Threshold
  threshold_type = type;
  threshold_value = value;
  /// Call the function to initialize
  threshold( src_gray_R, dst_R, threshold_value, max_BINARY_value,threshold_type );
  /// Wait until user finishes program
  result = dst_R;
}

Mat Left,Right;
int main()
{

Mat Image_Left_1,Image_Right_1;

source_image_L = imread("D:\\view4.png");											
source_image_R = imread("D:\\view5.png");	
Image_Left_1 = source_image_L;
Image_Right_1 = source_image_R;
Mat cut;
Mat cut_L,cut_R;

Mat im,im2;

Mat sub_L,sub_Ln,add_L,sub_R,sub_Rn,add_R;

pyrMeanShiftFiltering(Image_Right_1, Image_Right, 10, 10, 3); 
pyrMeanShiftFiltering(Image_Left_1, Image_Left, 10, 10, 3); 

imshow("Right", Image_Right);
imshow("Left", Image_Left);
waitKey(0);
destroyAllWindows();

medianBlur ( Image_Right, Image_Right_1,3);
medianBlur ( Image_Left, Image_Left_1,3);
GaussianBlur( Image_Right_1, Image_Right, Size( 3, 3 ), 0, 0 );
GaussianBlur( Image_Left_1, Image_Left, Size( 3, 3 ), 0, 0 );
cut = Image_Left;

//histogram_RGB_view(Image_Left);
cut_object_from_background_image_L(Image_Left,cut_L,10);
cut_object_from_background_image_R(Image_Right,cut_R,10);
imshow("Right", cut_R);
imshow("Left", cut_L);
waitKey(0);
destroyAllWindows();

sub_Ln = Image_Left - Image_Left_1;
sub_L = Image_Left_1 - Image_Left;
sub_Rn = Image_Right - Image_Right_1;
sub_R = Image_Right_1 - Image_Right;
add_L = sub_Ln + sub_L;
add_R = sub_R + sub_Rn;

Hold_R(add_R,1,0,Right);
Hold_L(add_L,1,0,Left);

im = Right;
im2 = Left;
medianBlur ( im, Right,3);
medianBlur ( im2, Left,3);

imshow("Right", Right);
imshow("Left", Left);
waitKey(0);
destroyAllWindows();
	
	src = cut;
// Perform the distance transform algorithm
    Mat dist;
    distanceTransform(Left, dist, CV_DIST_L2, 3);
    // Normalize the distance image for range = {0.0, 1.0}
    // so we can visualize and threshold it
    normalize(dist, dist, 0, 1., NORM_MINMAX);
    imshow("Distance Transform Image", dist);
	waitKey(0);
	destroyAllWindows();
	 // Threshold to obtain the peaks
    // This will be the markers for the foreground objects
    threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
    // Dilate a bit the dist image
    Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
    dilate(dist, dist, kernel1);
    imshow("Peaks", dist);
	waitKey(0);
	destroyAllWindows();
	  // Create the CV_8U version of the distance image
    // It is needed for findContours()
    Mat dist_8u;
    dist.convertTo(dist_8u, CV_8U);
    // Find total markers
    vector<vector<Point> > contours;
    findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    // Create the marker image for the watershed algorithm
    Mat markers = Mat::zeros(dist.size(), CV_32SC1);
    // Draw the foreground markers
    for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
    // Draw the background marker
    circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    imshow("Markers", markers*10000);
	waitKey(0);
	destroyAllWindows();

	 // Perform the watershed algorithm
    watershed(src, markers);
    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
//    imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                  // image looks like at that point
    // Generate random colors
    vector<Vec3b> colors;
    for (size_t i = 0; i < contours.size(); i++)
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
    // Create the result image
    Mat dst1 = Mat::zeros(markers.size(), CV_8UC3);
    // Fill labeled objects with random colors
    for (int i = 0; i < markers.rows; i++)
    {
        for (int j = 0; j < markers.cols; j++)
        {
            int index = markers.at<int>(i,j);
            if (index > 0 && index <= static_cast<int>(contours.size()))
                dst1.at<Vec3b>(i,j) = colors[index-1];
            else
                dst1.at<Vec3b>(i,j) = Vec3b(0,0,0);
        }
    }
    // Visualize the final image
    imshow("Final Result", dst1);
	waitKey(0);
	destroyAllWindows();


  
system("PAUSE");

return 0;
 
}
