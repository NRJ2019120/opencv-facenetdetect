// Summary: 使用OpenCV3.3.1中的face_detector
// Author: Amusi
// Date:   2018-02-28
// Reference: http://blog.csdn.net/minstyrain/article/details/78907425

#include <iostream>  
#include <cstdlib>  
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

using namespace cv;  
using namespace cv::dnn;  
using namespace std;  
const size_t inWidth = 300;  
const size_t inHeight = 300;  
const double inScaleFactor = 1.0;  
const Scalar meanVal(104.0, 177.0, 123.0);  
  
int main(int argc, char** argv)  
{  
    float min_confidence = 0.5;  
    String modelConfiguration = "face_detector/deploy.prototxt";  
    String modelBinary = "face_detector/res10_300x300_ssd_iter_140000.caffemodel";  
    //! [Initialize network]  
    dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);  
    //! [Initialize network]  
    if (net.empty())  
    {  
        cerr << "Can't load network by using the following files: " << endl;  
        cerr << "prototxt:   " << modelConfiguration << endl;  
        cerr << "caffemodel: " << modelBinary << endl;  
        cerr << "Models are available here:" << endl;  
        cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;  
        cerr << "or here:" << endl;  
        cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;  
        exit(-1);  
    }  
  
    VideoCapture cap(0);  
    if (!cap.isOpened())  
    {  
        cout << "Couldn't open camera : " << endl;  
        return -1;  
    }  
    for (;;)  
    {  
        Mat frame;  
        cap >> frame; // get a new frame from camera/video or read image  
  
        if (frame.empty())  
        {  
            waitKey();  
            break;  
        }  
  
        if (frame.channels() == 4)  
            cvtColor(frame, frame, COLOR_BGRA2BGR);  
  
        //! [Prepare blob]  
        Mat inputBlob = blobFromImage(frame, inScaleFactor,  
            Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images  
                                                             //! [Prepare blob]  
  
                                                             //! [Set input blob]  
        net.setInput(inputBlob, "data"); //set the network input  
                                         //! [Set input blob]  
  
                                         //! [Make forward pass]  
        Mat detection = net.forward("detection_out"); //compute output  
                                                      //! [Make forward pass]  
  
        vector<double> layersTimings;  
        double freq = getTickFrequency() / 1000;  
        double time = net.getPerfProfile(layersTimings) / freq;  
  
        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());  
  
        ostringstream ss;  
        ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";  
        putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));  
  
        float confidenceThreshold = min_confidence;  
        for (int i = 0; i < detectionMat.rows; i++)  
        {  
            float confidence = detectionMat.at<float>(i, 2);  
  
            if (confidence > confidenceThreshold)  
            {  
                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);  
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);  
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);  
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);  
  
                Rect object((int)xLeftBottom, (int)yLeftBottom,  
                    (int)(xRightTop - xLeftBottom),  
                    (int)(yRightTop - yLeftBottom));  
  
                rectangle(frame, object, Scalar(0, 255, 0));  
  
                ss.str("");  
                ss << confidence;  
                String conf(ss.str());  
                String label = "Face: " + conf;  
                int baseLine = 0;  
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);  
                rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),  
                    Size(labelSize.width, labelSize.height + baseLine)),  
                    Scalar(255, 255, 255), CV_FILLED);  
                putText(frame, label, Point(xLeftBottom, yLeftBottom),  
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));  
            }  
        }  
        cv::imshow("detections", frame);  
        if (waitKey(1) >= 0) break;  
    }  
    return 0;  
}  