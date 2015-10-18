#ifndef _LIB_CUT_OBJECT_FROM_BACKGROUND_H_
#define _LIB_CUT_OBJECT_FROM_BACKGROUND_H_

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
using namespace cv;
using namespace std;

void cut_object_from_background_image_L(Mat& image_L, Mat& result_cut_object_L, int border)
{
	
    int border2 = border + border;
    cv::Rect rectangle_image_L(border,border,image_L.cols-border2,image_L.rows-border2);
 
    Mat result_image_L; // segmentation result (4 possible values)
    Mat bgModel_image_L,fgModel_image_L; // the models (internally used)
 
    // GrabCut segmentation
    cv::grabCut(image_L,    // input image_L
        result_image_L,   // segmentation result
        rectangle_image_L,// rectangle containing foreground_image_L 
        bgModel_image_L,fgModel_image_L, // models
        1,        // number of iterations
        cv::GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground_image_L
    cv::compare(result_image_L,cv::GC_PR_FGD,result_image_L,cv::CMP_EQ);
    // Generate output image_L
    Mat foreground_image_L(image_L.size(),CV_8UC3,cv::Scalar(0,0,0));
    image_L.copyTo(foreground_image_L,result_image_L); // bg pixels not copied

	result_cut_object_L = foreground_image_L;

}

void cut_object_from_background_image_R(Mat& image_R, Mat& result_cut_object_R, int border)
{
	
    int border2 = border + border;
    cv::Rect rectangle_image_R(border,border,image_R.cols-border2,image_R.rows-border2);
 
    Mat result_image_R; // segmentation result (4 possible values)
    Mat bgModel_image_R,fgModel_image_R; // the models (internally used)
 
    // GrabCut segmentation
    cv::grabCut(image_R,    // input image_R
        result_image_R,   // segmentation result
        rectangle_image_R,// rectangle containing foreground_image_R 
        bgModel_image_R,fgModel_image_R, // models
        1,        // number of iterations
        cv::GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground_image_L
    cv::compare(result_image_R,cv::GC_PR_FGD,result_image_R,cv::CMP_EQ);
    // Generate output image_R
    Mat foreground_image_R(image_R.size(),CV_8UC3,cv::Scalar(0,0,0));
    image_R.copyTo(foreground_image_R,result_image_R); // bg pixels not copied

	result_cut_object_R = foreground_image_R;

}
#endif