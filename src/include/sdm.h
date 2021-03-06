#pragma once
#ifndef sdm_H_
#define sdm_H_

#include <iostream>
#include <vector>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "cereal/cereal.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal_extension/mat_cerealisation.hpp"

#include "helper.h"
#include "feature_descriptor.h"




#define SDM_NO_ERROR        0       //无错误
#define SDM_ERROR_FACEDET   200     //重新通过CascadeClassifier检测到人脸
#define SDM_ERROR_FACEPOS   201     //人脸位置变化较大，可疑
#define SDM_ERROR_FACESIZE  202     //人脸大小变化较大，可疑
#define SDM_ERROR_FACENO    203     //找不到人脸
#define SDM_ERROR_IMAGE     204     //图像错误

#define SDM_ERROR_ARGS      400     //参数传递错误
#define SDM_ERROR_MODEL     401     //模型加载错误



//回归器类
class LinearRegressor{

public:
    LinearRegressor();

    bool learn(cv::Mat &data, cv::Mat &labels, bool isPCA=false);

    double test(cv::Mat data, cv::Mat labels);

    cv::Mat predict(cv::Mat values);

    void convert(std::vector<int> &tar_LandmarkIndex);
private:
    cv::Mat weights;
    cv::Mat eigenvectors;
    cv::Mat meanvalue;
    cv::Mat x;
    bool isPCA;

    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(weights, meanvalue, x, isPCA);
        if(isPCA){
            ar(eigenvectors);
        }
    }
};


class sdm{

public:
    sdm();

    sdm(std::vector<std::vector<int>> LandmarkIndexs, std::vector<int> eyes_index, cv::Mat meanShape, std::vector<HoGParam> HoGParams, std::vector<LinearRegressor> LinearRegressors);


    void train(std::vector<ImageLabel> &mImageLabels);

    int detectPoint(const cv::Mat& src, cv::Mat& current_shape, cv::Rect faceBox);

    void printmodel();

    void convert(std::vector<int> &full_eyes_Indexs);

    cv::Vec3d EstimateHeadPose(cv::Mat &current_shape);

    void drawPose(cv::Mat& img, const cv::Mat& current_shape, cv::Vec3d pose, float lineL=50);

    float getPitch();
    float getRoll();
    float getYaw();
    std::vector<cv::Point2f> getPts();

private:
    cv::Rect faceBox;
    std::vector<std::vector<int>> LandmarkIndexs;
    std::vector<int> eyes_index;
    cv::Mat meanShape;
    float pitch_;
    float yaw_;
    float roll_;
    std::vector<HoGParam> HoGParams;
    bool isNormal;
    std::vector<LinearRegressor> LinearRegressors;
    cv::Mat estimateHeadPoseMat;
    cv::Mat estimateHeadPoseMat2;
    std::vector<cv::Point2f> pts_;
    int *estimateHeadPosePointIndexs;
    friend class cereal::access;
    /**
     * Serialises this class using cereal.
     *
     * @param[in] ar The archive to serialise to (or to serialise from).
     */
    template<class Archive>
    void serialize(Archive& ar)
    {
        ar(LandmarkIndexs, eyes_index, meanShape, HoGParams, isNormal, LinearRegressors);
    }
};

//加载模型
bool load_sdm(const std::string& modelString, sdm &model);

//保存模型
void save_sdm(sdm model, std::string filename);

#endif


