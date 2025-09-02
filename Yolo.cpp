//
// Created by 15680 on 2025/8/29.
//

#include "Yolo.h"
#include <fstream>
Yolo::Yolo()
{
}

Yolo::~Yolo()
{

}

bool Yolo::loadModel(const std::string& model,const std::string& classFile,const cv::Size size,
        float confidenceThreshold ,float nmsThreshold)
{
    model_ = cv::dnn::readNetFromONNX(model);
    model_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    model_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    modelShapeSize_ = size;
    confidenceThreshold_ = confidenceThreshold;
    nmsThreshold_ = nmsThreshold;

    std::ifstream inFile(classFile,std::ios::in);

    std::string className;
    while (std::getline(inFile,className))
    {
        class_names_.push_back(className);
    }
    return true;
}

cv::Mat Yolo::detect(const cv::Mat& src)
{
    cv::Size originSize = src.size();
    cv::Mat blob = preprocess(src);
    model_.setInput(blob);

    //
    std::vector<cv::Mat> outputs;
    model_.forward(outputs,model_.getUnconnectedOutLayersNames());

    cv::Mat output;
    if (outputs.size() == 1)
    {
        output = outputs[0];
    }
    else
    {
        output = outputs[1];
    }

    //
    std::vector<Object> objects = postprocess(output,originSize);

    //draw
    cv::Mat dst = src;
    drawObjects(dst,objects);
    return dst;
}

cv::Mat Yolo::preprocess(const cv::Mat& image)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(image,blob,1.0/255.0f,modelShapeSize_,cv::Scalar(0,0,0),true,false);
    return blob;
}

std::vector<Yolo::Object> Yolo::postprocess(const cv::Mat& output,const cv::Size& originSize)
{
    std::vector<Object> objects;
    const int dimensions = output.size[1];
    const int num_anchors = output.size[2];
    cv::Mat output_mat = output.reshape(1,dimensions);
    output_mat = output_mat.t();
    float x_factor = originSize.width * 1.0f / modelShapeSize_.width;
    float y_factor = originSize.height * 1.0f / modelShapeSize_.height;
    for (int i=0;i < num_anchors;i++)
    {
        cv::Mat scores = output_mat.row(i).colRange(4,dimensions);
        cv::Point class_id_points;
        double max_confidence;
        cv::minMaxLoc(scores,0,&max_confidence,0,&class_id_points);

        if (max_confidence > confidenceThreshold_)
        {
            float x = output_mat.at<float>(i,0);
            float y = output_mat.at<float>(i,1);
            float w = output_mat.at<float>(i,2);
            float h = output_mat.at<float>(i,3);

            int left = static_cast<int>((x - w/2) * x_factor );
            int top = static_cast<int>((y - h/2) *y_factor );
            int width = static_cast<int>(w * x_factor);
            int height = static_cast<int>(h * y_factor);

            Object object{};
            object.confidence = static_cast<float>(max_confidence);
            object.box = cv::Rect(left,top,width,height);
            object.color  = cv::Scalar(rand()% 255,rand()%255,rand() % 255);
            object.class_id = class_id_points.x;
            object.classify = (object.class_id < class_names_.size()) ?
                                  class_names_[object.class_id] : "unknown";

            objects.emplace_back(object);
        }
    }
    //nms
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const Object obj : objects)
    {
        boxes.push_back(obj.box);
        confidences.push_back(obj.confidence);
    }

    cv::dnn::NMSBoxes(boxes,confidences,confidenceThreshold_,nmsThreshold_,indices);

    std::vector<Object> finialObjects;
    for (int idx : indices)
    {
        finialObjects.push_back(objects[idx]);
    }

    return finialObjects;
}

void Yolo::drawObjects(cv::Mat& image,const std::vector<Object>& objections)
{
    for (const Object object : objections)
    {
        cv::rectangle(image,object.box,object.color,2);

        std::string label = object.classify + " : " + std::to_string(object.confidence);
        cv::putText(image,label,cv::Point(object.box.x,object.box.y-5),cv::FONT_HERSHEY_SIMPLEX,0.5f,cv::Scalar(0,255,0));
    }
}
