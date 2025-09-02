#include <iostream>
#include "Yolo.h"
#include <chrono>
class ElpasedTime
{
public:
    void start()
    {
        start_ =  std::chrono::high_resolution_clock::now();
    }
    int64_t elpasedTime()
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};
int main(int argc,char* argv[])
{
    Yolo yolo;
    if (!yolo.loadModel("../yolov8n.onnx","../cocos.name"))
    {
        throw std::runtime_error("load model failed!");
    }
    cv::Mat lean = cv::imread("D:/data/lena.jpg");
    ElpasedTime timer;
    timer.start();
    lean = yolo.detect(lean);
     std::cout << "detect time is " << timer.elpasedTime() << std::endl;
    cv::imshow("lena",lean);
    cv::waitKey(0);
    // cv::VideoCapture cap(0);
    //
    // if (cap.isOpened())
    // {
    //     cv::Mat frame;
    //     while (cap.read(frame))
    //     {
    //
    //         timer.start();
    //        frame = yolo.detect(frame);
    //         std::cout << "detect time is " << timer.elpasedTime() << std::endl;
    //         cv::imshow("cap",frame);
    //
    //         char key = cv::waitKey(27);
    //         if (key == 27)
    //         {
    //             break;
    //         }
    //     }
    //
    //     cap.release();
    // }
    return 0;
}