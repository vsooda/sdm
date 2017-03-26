#include <vector>
#include <iostream>
#include <fstream>


#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "sdm.h"

using namespace std;
using namespace cv;


std::string getFileString(const std::string& filepath) {
	std::ifstream is(filepath);
	std::string filebuffer="";
	if (is.is_open()) {
		// get length of file:
		is.seekg (0, is.end);
		long long length = is.tellg();
		is.seekg (0, is.beg);
		char * buffer = new char [length];
		std::cout << "Reading " << filepath << " " << length << " characters... ";
		// read data as a block:
		is.read (buffer,length);
		if (is)
			std::cout << "all characters read successfully." << std::endl;
		else
			std::cout << "error: only " << is.gcount() << " could be read";
		is.close();
		// ...buffer contains the entire file...
		filebuffer = std::string(buffer,length);
		delete[] buffer;
	} else {
		std::cout << filepath << "open faild in getFileString" << std::endl;
	}
	return filebuffer;
}

int main()
{
	/*********************
    std::vector<ImageLabel> mImageLabels;
    if(!load_ImageLabels("mImageLabels-test.bin", mImageLabels)){
        mImageLabels.clear();
        ReadLabelsFromFile(mImageLabels, "labels_ibug_300W_test.xml");
        save_ImageLabels(mImageLabels, "mImageLabels-test.bin");
    }
    std::cout << "测试数据一共有: " <<  mImageLabels.size() << std::endl;
	*******************/

    sdm modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    std::string modelString = getFileString(modelFilePath);
    while(!load_sdm(modelString, modelt)){
        std::cout << "文件打开错误，请重新输入文件路径." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if(!mCamera.isOpened()){
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat Image;
    cv::Mat current_shape;
    for(;;){
        mCamera >> Image;
        clock_t a = clock();
        modelt.track(Image, current_shape);
        std::cout << (clock() - a) * 1000.0 / CLOCKS_PER_SEC << std::endl;
        cv::Vec3d eav = modelt.EstimateHeadPose(current_shape);
        modelt.drawPose(Image, current_shape, eav, 50);

        int numLandmarks = current_shape.cols/2;
        for(int j=0; j<numLandmarks; j++){
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            std::stringstream ss;
            ss << j;
//            cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
            cv::circle(Image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }
        cv::imshow("Camera", Image);
        if(27 == cv::waitKey(5)){
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }

    system("pause");
    return 0;
}






















