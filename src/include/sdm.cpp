#include "sdm.h"
#include <sstream>

LinearRegressor::LinearRegressor() : weights(),meanvalue(),x(),isPCA(false)
{

}

bool LinearRegressor::learn(cv::Mat &data, cv::Mat &labels, bool isPCA)
{
    this->isPCA = isPCA;
    if(this->isPCA){
        cv::Mat mdata = data.colRange(0, data.cols-2).clone();
        cv::PCA FeaturePCA(mdata, cv::Mat(), CV_PCA_DATA_AS_ROW);
        std::cout << "特征向量尺度: " <<FeaturePCA.eigenvectors.size() << std::endl;
        std::cout << "特征值尺度:   " <<FeaturePCA.eigenvalues.size() << std::endl;
        double eigensum = cv::sum(FeaturePCA.eigenvalues)[0];
        double lamda = 0.0;
        int index = 0;
        for(int i=0; i<FeaturePCA.eigenvalues.rows; i++){
            lamda += FeaturePCA.eigenvalues.at<float>(i,0);
            if(lamda/eigensum > 0.97){
                index = i;
                std::cout << "特征个数可以压缩为:" << i << "个" << std::endl;
                break;
            }
        }
        this->meanvalue = FeaturePCA.mean;
        this->eigenvectors = FeaturePCA.eigenvectors.rowRange(0, index).t();
        for(int i=0; i<mdata.rows; i++){
            mdata.row(i) = mdata.row(i) - this->meanvalue;
        }
        mdata = mdata*this->eigenvectors;
        cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
        for(int i=0; i<mdata.rows; i++){
            for(int j=0; j<mdata.cols; j++){
                A.at<float>(i,j) = mdata.at<float>(i,j);
            }
        }
        A.col(A.cols-1) = cv::Mat::ones(A.rows, 1, A.type());
        mdata.release();
        //自己的写的最小二乘
        cv::Mat AT = A.t();
        cv::Mat ATA = A.t()*A;
        float lambda = 1.50f * static_cast<float>(cv::norm(ATA)) / static_cast<float>(A.rows);
        cv::Mat regulariser = cv::Mat::eye(ATA.size(), ATA.type())*lambda;
        regulariser.at<float>(regulariser.rows-1, regulariser.cols-1) = 0.0f;
        this->x = (ATA + regulariser).inv(cv::DECOMP_LU)*AT*labels;
        //opencv提供的最小二乘
        //cv::solve(A, labels, this->x);

//            this->weights = this->eigenvectors*this->x;
//            this->eigenvectors.release();
    }else{
        cv::Mat A = data.clone();
        //自己的写的最小二乘
        cv::Mat AT = A.t();
        cv::Mat ATA = A.t()*A;
        float lambda = 1.50f * static_cast<float>(cv::norm(ATA)) / static_cast<float>(A.rows);
        cv::Mat regulariser = cv::Mat::eye(ATA.size(), ATA.type())*lambda;
        regulariser.at<float>(regulariser.rows-1, regulariser.cols-1) = 0.0f;
        this->weights = (ATA + regulariser).inv(cv::DECOMP_LU)*AT*labels;
        //opencv提供的最小二乘
        //cv::solve(A, labels, this->weights);
    }
    return true; // see todo above
}

double LinearRegressor::test(cv::Mat data, cv::Mat labels)
{
    cv::Mat predictions;
    for (int i = 0; i < data.rows; ++i) {
        cv::Mat prediction = this->predict(data.row(i));
        predictions.push_back(prediction);
    }
    return cv::norm(predictions, labels, cv::NORM_L2) / cv::norm(labels, cv::NORM_L2);
}


cv::Mat LinearRegressor::predict(cv::Mat values)
{
    if(this->isPCA){
        cv::Mat mdata = values.colRange(0, values.cols-2).clone();
//            assert(mdata.cols==this->weights.rows && mdata.cols==this->meanvalue.cols);
        if(mdata.rows == 1){
            mdata = (mdata - this->meanvalue)*this->eigenvectors;
            cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
            for(int i=0; i<mdata.cols; i++){
                A.at<float>(i) = mdata.at<float>(i);
            }
            A.at<float>(A.cols-1) = 1.0f;
            return A*this->x;
        }
        else{
            for(int i=0; i<mdata.rows; i++){
                mdata.row(i) = mdata.row(i) - this->meanvalue;
            }
            mdata = mdata*this->eigenvectors;
            cv::Mat A = cv::Mat::zeros(mdata.rows, mdata.cols+1, mdata.type());
            for(int i=0; i<mdata.rows; i++){
                for(int j=0; j<mdata.cols; j++){
                    A.at<float>(i,j) = mdata.at<float>(i,j);
                }
            }
            A.col(A.cols-1) = cv::Mat::ones(A.rows, 1, A.type());
            return A*this->x;
        }
    }else{
        assert(values.cols==this->weights.rows);
        return  values*this->weights;
    }
}

void LinearRegressor::convert(std::vector<int> &tar_LandmarkIndex){
    if(isPCA){

    }else{
        assert(this->weights.cols/2 >= tar_LandmarkIndex.size());
        cv::Mat tmp = this->weights.clone();
        this->weights.release();
        this->weights.create(tmp.rows, tar_LandmarkIndex.size()*2, tmp.type());
        for(int i=0; i<this->weights.rows; i++){
            for(int j=0; j<tar_LandmarkIndex.size(); j++){
                this->weights.at<float>(i, j) = tmp.at<float>(i, tar_LandmarkIndex.at(j));
                this->weights.at<float>(i, j+tar_LandmarkIndex.size()) = tmp.at<float>(i, tar_LandmarkIndex.at(j)+tmp.cols/2);
            }
        }
        tmp.release();
    }
}

sdm::sdm(){
    //{36,39,42,45,30,48,54};   {7,16,17,8,9,10,11};
    //static int HeadPosePointIndexs[] = {36,39,42,45,30,48,54};
    static int HeadPosePointIndexs[] = {27, 29, 33, 31, 65, 46, 52};
    estimateHeadPosePointIndexs = HeadPosePointIndexs;
    static float estimateHeadPose2dArray[] = {
        -0.208764,-0.140359,0.458815,0.106082,0.00859783,-0.0866249,-0.443304,-0.00551231,-0.0697294,
        -0.157724,-0.173532,0.16253,0.0935172,-0.0280447,0.016427,-0.162489,-0.0468956,-0.102772,
        0.126487,-0.164141,0.184245,0.101047,0.0104349,-0.0243688,-0.183127,0.0267416,0.117526,
        0.201744,-0.051405,0.498323,0.0341851,-0.0126043,0.0578142,-0.490372,0.0244975,0.0670094,
        0.0244522,-0.211899,-1.73645,0.0873952,0.00189387,0.0850161,1.72599,0.00521321,0.0315345,
        -0.122839,0.405878,0.28964,-0.23045,0.0212364,-0.0533548,-0.290354,0.0718529,-0.176586,
        0.136662,0.335455,0.142905,-0.191773,-0.00149495,0.00509046,-0.156346,-0.0759126,0.133053,
        -0.0393198,0.307292,0.185202,-0.446933,-0.0789959,0.29604,-0.190589,-0.407886,0.0269739,
        -0.00319206,0.141906,0.143748,-0.194121,-0.0809829,0.0443648,-0.157001,-0.0928255,0.0334674,
        -0.0155408,-0.145267,-0.146458,0.205672,-0.111508,0.0481617,0.142516,-0.0820573,0.0329081,
        -0.0520549,-0.329935,-0.231104,0.451872,-0.140248,0.294419,0.223746,-0.381816,0.0223632,
        0.176198,-0.00558382,0.0509544,0.0258391,0.050704,-1.10825,-0.0198969,1.1124,0.189531,
        -0.0352285,0.163014,0.0842186,-0.24742,0.199899,0.228204,-0.0721214,-0.0561584,-0.157876,
        -0.0308544,-0.131422,-0.0865534,0.205083,0.161144,0.197055,0.0733392,-0.0916629,-0.147355,
        0.527424,-0.0592165,0.0150818,0.0603236,0.640014,-0.0714241,-0.0199933,-0.261328,0.891053};
    estimateHeadPoseMat = cv::Mat(15,9,CV_32FC1,estimateHeadPose2dArray);
    static float estimateHeadPose2dArray2[] = {
        0.139791,27.4028,7.02636,
        -2.48207,9.59384,6.03758,
        1.27402,10.4795,6.20801,
        1.17406,29.1886,1.67768,
        0.306761,-103.832,5.66238,
        4.78663,17.8726,-15.3623,
        -5.20016,9.29488,-11.2495,
        -25.1704,10.8649,-29.4877,
        -5.62572,9.0871,-12.0982,
        -5.19707,-8.25251,13.3965,
        -23.6643,-13.1348,29.4322,
        67.239,0.666896,1.84304,
        -2.83223,4.56333,-15.885,
        -4.74948,-3.79454,12.7986,
        -16.1,1.47175,4.03941 };
    estimateHeadPoseMat2 = cv::Mat(15,3,CV_32FC1,estimateHeadPose2dArray2);
}

sdm::sdm(std::vector<std::vector<int>> LandmarkIndexs, std::vector<int> eyes_index, cv::Mat meanShape, std::vector<HoGParam> HoGParams, std::vector<LinearRegressor> LinearRegressors) :
    LandmarkIndexs(LandmarkIndexs),eyes_index(eyes_index),meanShape(meanShape),HoGParams(HoGParams),isNormal(true),LinearRegressors(LinearRegressors)
{
}


void sdm::train(std::vector<ImageLabel> &mImageLabels){
    assert(HoGParams.size() >= LinearRegressors.size());
    int samplesNum = mImageLabels.size();
    std::cout << "一共" << samplesNum << "个训练样本.\n" << std::endl;

    isNormal = true;
    if(isNormal)
        std::cout << "归一化坐标.\n" << std::endl;
    else
        std::cout << "不归一化坐标.\n" << std::endl;

    cv::Mat current_shape(samplesNum, meanShape.cols, CV_32FC1);
    cv::Mat target_shape(samplesNum, meanShape.cols, CV_32FC1);
//        cv::namedWindow("Image", cv::WINDOW_NORMAL);
    for(int i=0; i<samplesNum; i++){
        for(int j=0; j<meanShape.cols; j++){
            target_shape.at<float>(i, j)  = mImageLabels.at(i).landmarkPos[j];
        }
        cv::Mat Image = cv::imread(mImageLabels.at(i).imagePath, CV_LOAD_IMAGE_GRAYSCALE); //CV_LOAD_IMAGE_GRAYSCALE
        cv::Rect faceBox(mImageLabels.at(i).faceBox[0],mImageLabels.at(i).faceBox[1],mImageLabels.at(i).faceBox[2],mImageLabels.at(i).faceBox[3]);
        cv::Rect efaceBox = get_enclosing_bbox(target_shape.row(i));
        cv::Rect mfaceBox = perturb(faceBox) & cv::Rect(0,0,Image.cols, Image.rows);
        if((float)(efaceBox & faceBox).area()/faceBox.area()<0.4)
            mfaceBox = perturb(efaceBox) & cv::Rect(0,0,Image.cols, Image.rows);
        cv::Mat  align_shape = align_self_mean(meanShape, mfaceBox);
        assert(align_shape.rows == 1);
        for(int j=0; j<meanShape.cols; j++){
            current_shape.at<float>(i, j) = align_shape.at<float>(j);
        }
    }
    float error0 = 0.0f;
    int numLandmarks = target_shape.cols/2;
    for(int i=0; i<samplesNum; i++){
        cv::Mat shape = current_shape.row(i);
        float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
        float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
        float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
        float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
        float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );//计算两眼的距离
        for(int j=0; j<numLandmarks; j++){
            float dx = target_shape.at<float>(i, j) - current_shape.at<float>(i, j);
            float dy = target_shape.at<float>(i, j+numLandmarks) - current_shape.at<float>(i, j+numLandmarks);
            error0 += sqrt(dx*dx + dy*dy)/distance;
        }
    }
    error0 = error0/samplesNum/numLandmarks;
    std::cout <<"初始误差为: " << error0 << "\n" << std::endl;


    for(int i=0; i<LinearRegressors.size(); i++){
        //开始计算描述子
        int bins = 1;
        for(int j=0; j<HoGParams.at(i).num_bins; j++)
            bins = 2*bins;
        cv::Mat HogDescriptors(samplesNum, (bins*HoGParams.at(i).num_cells*HoGParams.at(i).num_cells)*LandmarkIndexs.at(i).size()+1, CV_32FC1);
        for(int j=0; j<samplesNum; j++){
            cv::Mat grayImage = cv::imread(mImageLabels.at(j).imagePath, CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape.row(j), LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
            assert(Descriptor.cols == HogDescriptors.cols);
            for(int k=0; k<Descriptor.cols; k++){
                HogDescriptors.at<float>(j, k) = Descriptor.at<float>(0,k);
            }
        }
        //描述子计算完成，开始一次迭代
        cv::Mat update_step = target_shape - current_shape;
        int numLandmarks = update_step.cols/2;
        if(isNormal){
            for(int j=0; j<samplesNum; j++){
                cv::Mat shape = current_shape.row(j);
                float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
                float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
                float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
                float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
                float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
                update_step.row(j) = update_step.row(j)/distance;
            }
        }
        LinearRegressors.at(i).learn(HogDescriptors, update_step);
        update_step = LinearRegressors.at(i).predict(HogDescriptors);
        if(isNormal){
            for(int j=0; j<samplesNum; j++){
                cv::Mat shape = current_shape.row(j);
                float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
                float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
                float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
                float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
                float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
                update_step.row(j) = update_step.row(j)*distance;
            }
        }
        current_shape = current_shape + update_step;
        //一次迭代结束，更新梯度变化，计算误差

        float error = 0.0f;
        for(int i=0; i<samplesNum; i++){
            cv::Mat shape = current_shape.row(i);
            float lx = ( shape.at<float>(eyes_index.at(0))+shape.at<float>(eyes_index.at(1)) )*0.5;
            float ly = ( shape.at<float>(eyes_index.at(0)+numLandmarks)+shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
            float rx = ( shape.at<float>(eyes_index.at(2))+shape.at<float>(eyes_index.at(3)) )*0.5;
            float ry = ( shape.at<float>(eyes_index.at(2)+numLandmarks)+shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
            float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );//计算两眼的距离
            for(int j=0; j<numLandmarks; j++){
                float dx = target_shape.at<float>(i, j) - current_shape.at<float>(i, j);
                float dy = target_shape.at<float>(i, j+numLandmarks) - current_shape.at<float>(i, j+numLandmarks);
                error += sqrt(dx*dx + dy*dy)/distance;
            }
        }
        error = error/samplesNum/numLandmarks;
        std::cout << "现在平均误差是: " << error << "\n" << std::endl;
    }
}

int sdm::detectPoint(const cv::Mat& src, cv::Mat& current_shape, cv::Rect faceBox){
    cv::Mat grayImage;
    if(src.channels() == 1){
        grayImage = src;
    }else if(src.channels() == 3){
        cv::cvtColor(src, grayImage, CV_BGR2GRAY);
    }else if(src.channels() == 4){
        cv::cvtColor(src, grayImage, CV_RGBA2GRAY);
    }

    current_shape = align_self_mean(meanShape, faceBox);
    int numLandmarks = current_shape.cols/2;
    for(int i=0; i<LinearRegressors.size(); i++){
        cv::Mat Descriptor = CalculateHogDescriptor(grayImage, current_shape, LandmarkIndexs.at(i), eyes_index, HoGParams.at(i));
        cv::Mat update_step = LinearRegressors.at(i).predict(Descriptor);
        if(isNormal){
            float lx = ( current_shape.at<float>(eyes_index.at(0))+current_shape.at<float>(eyes_index.at(1)) )*0.5;
            float ly = ( current_shape.at<float>(eyes_index.at(0)+numLandmarks)+current_shape.at<float>(eyes_index.at(1)+numLandmarks) )*0.5;
            float rx = ( current_shape.at<float>(eyes_index.at(2))+current_shape.at<float>(eyes_index.at(3)) )*0.5;
            float ry = ( current_shape.at<float>(eyes_index.at(2)+numLandmarks)+current_shape.at<float>(eyes_index.at(3)+numLandmarks) )*0.5;
            float distance = sqrt( (rx-lx)*(rx-lx)+(ry-ly)*(ry-ly) );
            update_step = update_step*distance;
        }
        current_shape = current_shape + update_step;
    }

    pts_.clear();
    for(int j=0; j<numLandmarks; j++){
        int x = current_shape.at<float>(j);
        int y = current_shape.at<float>(j + numLandmarks);
        pts_.push_back(cv::Point2f(x, y));
    }
        
    return 0;
}

std::vector<cv::Point2f> sdm::getPts() {
    return pts_;
}

void sdm::printmodel(){
    if(isNormal)
        std::cout << "以两眼距离归一化步长" << std::endl;
    else
        std::cout << "不归一化步长" << std::endl;
    std::cout << "一共" << LinearRegressors.size() << "次迭代回归..." << std::endl;
    for(int i=0; i<LandmarkIndexs.size(); i++){
        std::cout <<"第"<<i<<"次回归: "<<LandmarkIndexs.at(i).size()<<"个点  ";
        std::cout << "num_cells:"<<HoGParams.at(i).num_cells<<"  cell_size:"<<HoGParams.at(i).cell_size <<"  num_bins:"<<HoGParams.at(i).num_bins<<"  relative_patch_size:"<<HoGParams.at(i).relative_patch_size<<std::endl;
    }
}

void sdm::convert(std::vector<int> &full_eyes_Indexs){
    std::vector<int> tar_LandmarkIndex;
    for(int i=0; i<LinearRegressors.size(); i++){
        for(int j=0; j<LandmarkIndexs.at(i).size(); j++){
            int t = LandmarkIndexs.at(i).at(j);
            bool flag = true;
            for(int k=0; k<tar_LandmarkIndex.size(); k++){
                if(t == tar_LandmarkIndex.at(k)){
                    flag = false;
                    break;
                }
            }
            if(flag)
                tar_LandmarkIndex.push_back(t);
        }
    }
    for(int i=0; i<full_eyes_Indexs.size(); i++){
        int t = full_eyes_Indexs.at(i);
        bool flag = true;
        for(int j=0; j<tar_LandmarkIndex.size(); j++){
            if(t == tar_LandmarkIndex.at(j)){
                flag = false;
                break;
            }
        }
        if(flag)
            tar_LandmarkIndex.push_back(t);
    }
    //更新转换meanShape
    cv::Mat tmp = meanShape.clone();
    meanShape.release();
    meanShape.create(1, tar_LandmarkIndex.size()*2, tmp.type());
    for(int i=0; i<tar_LandmarkIndex.size(); i++){
        meanShape.at<float>(i) = tmp.at<float>(tar_LandmarkIndex.at(i));
        meanShape.at<float>(i+tar_LandmarkIndex.size()) = tmp.at<float>(tar_LandmarkIndex.at(i)+tmp.cols/2);
    }
    //更新转换LandmarkIndexs
    for(int i=0; i<LinearRegressors.size(); i++){
        for(int j=0; j<LandmarkIndexs.at(i).size(); j++){
            for(int k=0; k<tar_LandmarkIndex.size(); k++){
                if(LandmarkIndexs.at(i).at(j) == tar_LandmarkIndex.at(k)){
                    LandmarkIndexs.at(i).at(j) = k;
                    break;
                }
            }
        }
    }
    //更新转换eyes_index
    for(int i=0; i<eyes_index.size(); i++){
        bool flag = false;
        for(int j=0; i<tar_LandmarkIndex.size(); j++){
            if(eyes_index.at(i) == tar_LandmarkIndex.at(j)){
                eyes_index.at(i) = j;
                flag = true;
                break;
            }
        }
        assert(flag);
    }
    //更新转换LinearRegressors
    for(int i=0; i<LinearRegressors.size(); i++){
        LinearRegressors.at(i).convert(tar_LandmarkIndex);
    }
}

cv::Vec3d sdm::EstimateHeadPose(cv::Mat &current_shape) {
    if(current_shape.empty())
        return cv::Vec3d(0, 0, 0);
    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
    cv::Mat predict = tmp*estimateHeadPoseMat2;
    cv::Vec3d eav;
    eav[0] = predict.at<float>(0);
    eav[1] = predict.at<float>(1);
    eav[2] = predict.at<float>(2);
    pitch_ = eav[0];
    yaw_ = eav[1];
    roll_ = eav[2];
    return eav;
}

float sdm::getPitch() {
    return pitch_;
}

float sdm::getYaw() {
    return yaw_;
}

float sdm::getRoll() {
    return roll_;
}

void sdm::drawPose(cv::Mat& img, const cv::Mat& current_shape, cv::Vec3d pose, float lineL)
{
    if(current_shape.empty())
        return;

    std::string txt, txt1, txt2;
    std::stringstream ss3;
    ss3 << pose[0];
    txt = "Pitch: " + ss3.str();
    cv::putText(img, txt,  cv::Point(340, 20), 0.5,0.5, cv::Scalar(255,255,255));
    std::stringstream ss4;
    ss4 << pose[1];
    txt1 = "Yaw: " + ss4.str();
    cv::putText(img, txt1, cv::Point(340, 40), 0.5,0.5, cv::Scalar(255,255,255));
    std::stringstream ss5;
    ss5 << pose[2];
    txt2 = "Roll: " + ss5.str();
    cv::putText(img, txt2, cv::Point(340, 60), 0.5,0.5, cv::Scalar(255,255,255));

    static const int samplePdim = 7;
    float miny = 10000000000.0f;
    float maxy = 0.0f;
    float sumx = 0.0f;
    float sumy = 0.0f;
    for(int i=0; i<samplePdim; i++){
        sumx += current_shape.at<float>(estimateHeadPosePointIndexs[i]);
        float y = current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2);
        sumy += y;
        if(miny > y)
            miny = y;
        if(maxy < y)
            maxy = y;
    }
    float dist = maxy - miny;
    sumx = sumx/samplePdim;
    sumy = sumy/samplePdim;
    static cv::Mat tmp(1, 2*samplePdim+1, CV_32FC1);
    for(int i=0; i<samplePdim; i++){
        tmp.at<float>(i) = (current_shape.at<float>(estimateHeadPosePointIndexs[i])-sumx)/dist;
        tmp.at<float>(i+samplePdim) = (current_shape.at<float>(estimateHeadPosePointIndexs[i]+current_shape.cols/2)-sumy)/dist;
    }
    tmp.at<float>(2*samplePdim) = 1.0f;
    cv::Mat predict = tmp*estimateHeadPoseMat;
    cv::Mat rot(3,3,CV_32FC1);
    for(int i=0; i<3; i++){
        rot.at<float>(i,0) = predict.at<float>(3*i);
        rot.at<float>(i,1) = predict.at<float>(3*i+1);
        rot.at<float>(i,2) = predict.at<float>(3*i+2);
    }
    //we have get the rot mat
    int loc[2] = {70, 70};
    int thickness = 2;
    int lineType  = 8;

    cv::Mat P = (cv::Mat_<float>(3,4) <<
        0, lineL, 0,  0,
        0, 0, -lineL, 0,
        0, 0, 0, -lineL);
    P = rot.rowRange(0,2)*P;
    P.row(0) += loc[0];
    P.row(1) += loc[1];
    cv::Point p0(P.at<float>(0,0),P.at<float>(1,0));

    line(img, p0, cv::Point(P.at<float>(0,1),P.at<float>(1,1)), cv::Scalar( 255, 0, 0 ), thickness, lineType);
    line(img, p0, cv::Point(P.at<float>(0,2),P.at<float>(1,2)), cv::Scalar( 0, 255, 0 ), thickness, lineType);
    line(img, p0, cv::Point(P.at<float>(0,3),P.at<float>(1,3)), cv::Scalar( 0, 0, 255 ), thickness, lineType);
}

//加载模型
bool load_sdm(const std::string& modelString, sdm &model)
{
    std::cout << "load from string" << std::endl;
    std::istringstream model_sin(modelString);
    cereal::BinaryInputArchive input_archive(model_sin);
    input_archive(model);
    return true;
}

//保存模型
void save_sdm(sdm model, std::string filename)
{
    std::ofstream file(filename, std::ios::binary);
    cereal::BinaryOutputArchive output_archive(file);
    output_archive(model);
    file.close();
}


