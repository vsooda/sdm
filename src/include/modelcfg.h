#ifndef MODELCFG_H_
#define MODELCFG_H_

#define LandmarkPointsNum  74

static char trainFilePath[] = "ibug_300W_large_face_landmark_dataset/";    //   F:/BaiduYunDownload/ibug_300W_large_face_landmark_dataset/

static float mean_norm_shape[] = {0.0645445,0.0692465,0.0945456,0.13893,0.209027,0.296759,0.399217,0.521555,0.64216,0.746397,0.835007,0.903347,0.947881,0.971616,0.97556,0.904567,0.821804,0.701432,0.622545,0.708196,0.820383,0.130368,0.209076,0.331437,0.409337,0.324029,0.211921,0.221673,0.304391,0.392713,0.300227,0.814411,0.734392,0.642233,0.73804,0.451012,0.439318,0.381262,0.42054,0.51871,0.618966,0.654031,0.596446,0.584201,0.443572,0.589706,0.346885,0.406539,0.468892,0.519363,0.569618,0.633212,0.691628,0.645445,0.592356,0.52109,0.450483,0.396336,0.442758,0.5197,0.595938,0.594892,0.519512,0.44383,0.519677,0.518887,0.256496,0.2535,0.349415,0.353902,0.682835,0.687206,0.783858,0.781808,0.291022,0.440775,0.588375,0.732721,0.846148,0.938059,1.01476,1.04029,1.01431,0.936887,0.844194,0.73107,0.583509,0.436284,0.287268,0.185557,0.116374,0.126077,0.172773,0.176754,0.168304,0.187633,0.119434,0.127479,0.174658,0.177108,0.171355,0.292883,0.260645,0.313222,0.326059,0.291226,0.258371,0.311548,0.324102,0.309262,0.43445,0.557249,0.609424,0.623466,0.607487,0.556525,0.434639,0.310318,0.585149,0.585447,0.749746,0.727112,0.710779,0.719877,0.710263,0.726101,0.748126,0.799773,0.832611,0.843221,0.834142,0.801871,0.772548,0.784307,0.772173,0.752387,0.757743,0.752951,0.770764,0.531223,0.266325,0.318312,0.321365,0.274446,0.272206,0.319463,0.31639,0.264766};
//0 1 2 14 15 16这几个点索引一般不用


//需要计算特征的landmark索引，通过一下12个点的HOG特征回归出68个点的梯度变化
static int LandmarkLength1 = 12;
static int IteraLandmarkIndex1[] = {2, 5, 7, 9, 12, 22, 16, 27, 31, 65, 46, 52};
static int LandmarkLength2 = 12;
static int IteraLandmarkIndex2[] = {2, 5, 7, 9, 12, 22, 16, 27, 31, 65, 46, 52};
static int LandmarkLength3 = 12;
static int IteraLandmarkIndex3[] = {2, 5, 7, 9, 12, 22, 16, 27, 31, 65, 46, 52};
static int LandmarkLength4 = 20;
static int IteraLandmarkIndex4[] = {2, 5, 7, 9, 12, 22, 23, 17, 16, 27, 29,  33, 31, 65, 37,41, 46, 39, 52, 55};
static int LandmarkLength5 = 20;
static int IteraLandmarkIndex5[] = {2, 5, 7, 9, 12, 22, 23, 17, 16, 27, 29,  33, 31, 65, 37,41, 46, 39, 52, 55};


//四次迭代
//int LandmarkLength1 = 12;
//int IteraLandmarkIndex1[] = {3, 6, 8, 10, 13, 19, 24, 36, 45, 30, 48, 54};
//int LandmarkLength2 = 20;
//int IteraLandmarkIndex2[] = {3, 6, 8, 10, 13, 18, 20, 23, 25, 36, 39, 42, 45, 30, 31, 35, 48, 51, 54, 57};
//int LandmarkLength3 = 20;
//int IteraLandmarkIndex3[] = {3, 6, 8, 10, 13, 18, 20, 23, 25, 36, 39, 42, 45, 30, 31, 35, 48, 51, 54, 57};
//int LandmarkLength4 = 38;
//int IteraLandmarkIndex4[] = {0,2,4,6,8,10,12,14,16,17,19,21,22,24,26,27,30,31,33,35,36,37,39,40,42,43,45,46,48,50,52,54,56,58,60,62,64,66};


//左右两只眼睛的索引
static int eyes_indexs[4] = {27, 29, 33, 31};

static int extern_point_Length = 14;
static int extern_point_indexs[] = {0,16,36,37,38,39,40,41,42,43,44,45,46,47};

////5次68个点的迭代
//int LandmarkLength1 = 68;
//int IteraLandmarkIndex1[68] = {0};
//int LandmarkLength2 = 68;
//int IteraLandmarkIndex2[68] = {0};
//int LandmarkLength3 = 68;
//int IteraLandmarkIndex3[68] = {0};
//int LandmarkLength4 = 68;
//int IteraLandmarkIndex4[68] = {0};
//int LandmarkLength5 = 68;
//int IteraLandmarkIndex5[68] = {0};



//左右眼睛的四个眼角，鼻尖点，左右两个嘴角，一共七个点，估计头部姿态，2*7+1维度的数据，最小二乘拟合出9个维度的数据，即一个3*3的旋转矩阵
//矩阵大小为15*9
static float estimateHeadPoseMat[] = {  -0.258801,-0.142125,0.445513,0.101524,-0.0117096,-0.119747,-0.426367,-0.0197618,-0.143073,
                                 -0.194121,-0.210882,0.0989902,0.0822748,-0.00544055,0.0184441,-0.0628809,-0.0944775,-0.162363,
                                 0.173311,-0.205982,0.105287,0.0767408,0.0101697,0.0156599,-0.0632534,0.0774872,0.139928,
                                 0.278776,-0.109497,0.537723,0.0488799,0.00548235,0.111033,-0.471475,0.0280982,0.157491,
                                 0.0427104,-0.348899,-1.95092,0.0493076,0.0340635,0.157101,2.01808,-0.0716708,0.0860774,
                                 -0.191908,0.551951,0.456261,-0.174833,-0.0202239,-0.203346,-0.575386,0.105571,-0.171957,
                                 0.150051,0.465426,0.307133,-0.183886,-0.0123275,0.0208533,-0.4187,-0.0252474,0.0939203,
                                 0.00521464,0.229863,0.0595028,-0.480886,-0.0684972,0.43404,-0.0206778,-0.428706,0.118848,
                                 0.0125229,0.140842,0.115793,-0.239542,-0.0933311,0.0913729,-0.106839,-0.0523733,0.0697435,
                                 0.030548,-0.101407,-0.0659365,0.220726,-0.113126,0.0189044,0.0785027,-0.02235,0.0964722,
                                 0.0143054,-0.274282,-0.173696,0.477843,-0.073234,0.297015,0.180833,-0.322039,0.0855057,
                                 0.117061,-0.00704583,0.0157153,0.00142929,-0.106156,-1.29549,-0.0134561,1.22806,0.048107,
                                 -0.0663207,0.0996722,0.0374666,-0.215455,0.240434,0.233645,-0.0148478,-0.144342,-0.175324,
                                 -0.113332,-0.0876358,0.011164,0.23588,0.213911,0.2205,-0.103526,-0.258239,-0.243352,
                                 0.535077,0.000906855,-0.0336819,0.015495,0.586095,-0.14663,0.0643886,-0.114478,0.937324};







#endif
