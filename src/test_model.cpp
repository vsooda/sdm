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

bool track(const cv::Mat& src, cv::Mat& current_shape, cv::Rect& faceBox, cv::CascadeClassifier face_cascade) {
    cv::Mat grayImage;
    if(src.channels() == 1){
        grayImage = src;
    }else if(src.channels() == 3){
        cv::cvtColor(src, grayImage, CV_BGR2GRAY);
    }else if(src.channels() == 4){
        cv::cvtColor(src, grayImage, CV_RGBA2GRAY);
    }

    if(!current_shape.empty()){
        faceBox = get_enclosing_bbox(current_shape);
    }else{
        faceBox = cv::Rect(0,0,0,0);
    }
    int error_code = SDM_NO_ERROR;
    cv::Rect mfaceBox = faceBox & cv::Rect(0, 0, grayImage.cols, grayImage.rows);
    float ratio = ((float)faceBox.width)/faceBox.height;
    if(faceBox.area()<100){
        std::vector<cv::Rect> mFaceRects;
        face_cascade.detectMultiScale(grayImage, mFaceRects, 1.3, 3, 0, cv::Size(100, 100));
        if(mFaceRects.size() <=0){
            current_shape = cv::Mat();
            return false;
        }
        faceBox = mFaceRects[0];
        for(int i=1; i<mFaceRects.size(); i++){
            if(faceBox.area() < mFaceRects[i].area())
                faceBox = mFaceRects[i];
        }
    }
    return true;
}

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

int main(int argc, char** argv)
{
    cv::CascadeClassifier face_cascade;
    std::string faceModel("haar_roboman_ff_alt2.xml");
    face_cascade.load(faceModel);
    if(face_cascade.empty()){
        std::cout << "人脸检测模型加载失败." << std::endl;
    }

    sdm modelt;
    std::string modelFilePath = "self_sdm_model.bin";
    std::string modelString = getFileString(modelFilePath);
    while(!load_sdm(modelString, modelt)){
        std::cout << "文件打开错误，请重新输入文件路径." << std::endl;
        std::cin >> modelFilePath;
    }

    bool testVideo = false;

    if (!testVideo) {
        //测试单张图片
        for (int index = 4; index < argc; index++) {
            //Mat image = cv::imread("examples/data/ibug_lfpw_trainset/image_0005.png");
			std::string filename(argv[index]);
			std::cout << filename << std::endl;
            cv::Mat image = cv::imread(filename);
            cv::Mat current_shape;
            vector<cv::Rect> detected_faces;
            face_cascade.detectMultiScale(image, detected_faces, 1.2, 2, 0, cv::Size(50, 50));
            if (detected_faces.size() > 0) {
                cv::Rect rect = detected_faces[0];
                modelt.detectPoint(image, current_shape, rect);
                cv::Vec3d eav = modelt.EstimateHeadPose(current_shape);
                modelt.drawPose(image, current_shape, eav, 50);
                std::vector<cv::Point2f> pts = modelt.getPts();

                for(int j=0; j < pts.size(); j++){
                    cv::circle(image, pts[j], 2, cv::Scalar(0, 0, 255), -1);
                }
                cv::imshow("result", image);
                cv::waitKey();
            }
        }
        return 0;
    }

    cv::VideoCapture mCamera(0);
    if(!mCamera.isOpened()){
        std::cout << "Camera opening failed..." << std::endl;
        return 0;
    }
    cv::Mat Image;
    cv::Mat current_shape;
    for(;;){
        mCamera >> Image;
        clock_t a = clock();
        cv::Rect rect;
        bool isface = track(Image, current_shape, rect, face_cascade);
        if (isface) {
            modelt.detectPoint(Image, current_shape, rect);
            std::cout << (clock() - a) * 1000.0 / CLOCKS_PER_SEC << std::endl;
            cv::Vec3d eav = modelt.EstimateHeadPose(current_shape);
            modelt.drawPose(Image, current_shape, eav, 50);

            std::vector<cv::Point2f> pts = modelt.getPts();

            for(int j=0; j < pts.size(); j++){
                //cv::putText(Image, ss.str(), cv::Point(x, y), 0.5, 0.5, cv::Scalar(0, 0, 255));
                cv::circle(Image, pts[j], 2, cv::Scalar(0, 0, 255), -1);
            }
        }
        cv::imshow("Camera", Image);
        if(27 == cv::waitKey(5)){
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }
    return 0;
}






















