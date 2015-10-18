#ifndef _LIB_DATA_H_
#define _LIB_DATA_H_

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
#include <math.h>
#include <fstream>
#include <string.h>
#include <time.h>
#include <cstdlib>

using namespace cv;
using namespace std;


unsigned char vector_value[15][15] =
{
0	,1	,2	,3	,4	,5	,6	,7	,8	,9	,10	,11	,12	,13	,14
,15	,16	,17	,18	,19	,20	,21	,22	,23	,24	,25	,26	,27	,28	,29
,30	,31	,32	,33	,34	,35	,36	,37	,38	,39	,40	,41	,42	,43	,44
,45	,46	,47	,48	,49	,50	,51	,52	,53	,54	,55	,56	,57	,58	,59
,60	,61	,62	,63	,64	,65	,66	,67	,68	,69	,70	,71	,72	,73	,74
,75	,76	,77	,78	,79	,80	,81	,82	,83	,84	,85	,86	,87	,88	,89
,90	,91	,92	,93	,94	,95	,96	,97	,98	,99	,100	,101	,102	,103	,104
,105	,106	,107	,108	,109	,110	,111	,112	,113	,114	,115	,116	,117	,118	,119
,120	,121	,122	,123	,124	,125	,126	,127	,128	,129	,130	,131	,132	,133	,134
,135	,136	,137	,138	,139	,140	,141	,142	,143	,144	,145	,146	,147	,148	,149
,150	,151	,152	,153	,154	,155	,156	,157	,158	,159	,160	,161	,162	,163	,164
,165	,166	,167	,168	,169	,170	,171	,172	,173	,174	,175	,176	,177	,178	,179
,180	,181	,182	,183	,184	,185	,186	,187	,188	,189	,190	,191	,192	,193	,194
,195	,196	,197	,198	,199	,200	,201	,202	,203	,204	,205	,206	,207	,208	,209
,210	,211	,212	,213	,214	,215	,216	,217	,218	,219	,220	,221	,222	,223	,224
};
int vector_X_value[225] =
{
-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
,-7	,-6	,-5	,-4	,-3	,-2	,-1	,0	,1	,2	,3	,4	,5	,6	,7
};
int vector_Y_value[255] = 
{
-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,	-7,
-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,	-6,
-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,	-5,
-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,	-4,
-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,	-3,
-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,	-2,
-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,	-1,
0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,
1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,	1,
2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,	2,
3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,	3,
4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,	4,
5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,	5,
6,	6,	6,	6,	6,	6,	6,	6,	6,	6,	6,	6,	6,	6,	6,
7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7,	7
};

void ghep(int dx,int dy,int &value)
{
	int x,y;
	x=dx+7;
	y=dy+7;
	value=vector_value[y][x];
	
}

struct Motion
{
	int Pos_X;
	int Pos_Y;
	int DX;
	int DY;	
	int Use_I;
	double PSNR;
	int value;
	unsigned char value2;
} ;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat Image_Left;
Mat Image_Right;
Mat Block_Left;
Mat Block_Right;

Mat src, src_gray;
Mat dst, detected_edges;

#endif