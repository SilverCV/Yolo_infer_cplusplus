//
// Created by 15680 on 2025/8/29.
//

#ifndef YOLO_YOLO_H
#define YOLO_YOLO_H
#include <opencv2/opencv.hpp>
class Yolo
{
public:
    struct Object
    {
        int class_id;
        std::string classify;
        float confidence;
        cv::Scalar color;
        cv::Rect box;
    };
    Yolo();
    ~Yolo();

    bool loadModel(const std::string& model,const std::string& ,const cv::Size size = cv::Size(640,640),
        float confidenceThreshold = 0.4f,float nmsThreshold = 0.5f);



    cv::Mat detect(const cv::Mat& src);
protected:
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<Object> postprocess(const cv::Mat& output,const cv::Size& originSize);
    void drawObjects(cv::Mat& image,const std::vector<Object>& objections);
private:
    cv::dnn::Net model_;
    cv::Size modelShapeSize_;
    float confidenceThreshold_;
    float nmsThreshold_;

    std::vector<std::string> class_names_;
};


#endif //YOLO_YOLO_H