/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: Jovanie
 *
 * Created on February 16, 2018, 1:06 AM
 */

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/video/video.hpp>
#include<iostream>
#include<conio.h>           
#include "Blob.h"
#include "Timer.h"
//#define SHOW_STEPS   
#include<ctime>
#include <time.h>
#include <stdio.h>
#include <stack>


using namespace std;
using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);
const cv::Scalar SCALAR_AQUA = cv::Scalar(255.0,255.0,0.0);


// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, cv::Mat &imgFrame2Copy);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition,int &intVerticalLinePosition, int &carCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy,std::vector<Blob> &currentFrameBlob,double &frameCounter, int &fps);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
bool carOnFirstLane(vector<Blob> &blobs, int &intFirstLane);
bool carOnSecondLane(vector<Blob>& blobs, int& intSecondLane);
///////////////////////////////////////////////////////////////////////////////////////////////////



//return converted  int into a string 
string intToString(int number){
    std::stringstream ss;
    ss<<number;
    return ss.str();
}



//return the current date && time of the local machine
string getDateTime(){
    time_t now = time(0);
    char* dt = ctime(&now);
    string currentDate;
    currentDate = dt;
    return currentDate;
    
}


//timer
int getTimerSeconds(){
    time_t now;
    struct tm nowLocal;
    localtime(&now);
    nowLocal = *localtime(&now);
    int curSeconds = nowLocal.tm_sec;
    
    return curSeconds;
}






int main(void) {

   cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    vector<Blob> blobs;

    
    
    //TODO variables for 
    Point crossingLine[5];
    Point crossingLineDemo[5];
    
    Point dFirstLine[10];
    Point dSecondLine[10];

    //TODO variables for the timers 
    int carCount = 0;
    
    Timer timer1;
    Timer timer2;
    double totalTick;
    
    //Counter for Picture
    int picNum = 0;
    
    
    //TODO Kalman Filtering//
    int stateSize = 6;
    int meaSize = 4;
    int contrSize = 0;
    
    unsigned int type = CV_32F;
    cv::KalmanFilter kf(stateSize, meaSize, contrSize, type);
    
    
    
    cv::Mat state(stateSize,1, type);
    cv::Mat meas(meaSize, 1, type);
    
    //Transition State Matrix A
    cv::setIdentity(kf.transitionMatrix);
    
    
    //Measure Matrix H
    cv::setIdentity(kf.transitionMatrix);
    kf.measurementMatrix = cv::Mat::zeros(meaSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;
    
    // Process Noise Covariance Matrix Q
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
    
    // Measures Noise Covariance Matrix R
    cv::setIdentity(kf.measurementNoiseCov,cv::Scalar(1e-1));

//    std::vector<Blob> blobs;
    
    capVideo.open("v5.mp4");

    if (!capVideo.isOpened()) {                                                 // if unable to open video file
        std::cout << "error reading video file" << std::endl << std::endl;      // show error message
        _getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);                                                              // and exit program
    }

    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "error: video file must have at least two frames";
        _getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);
    }
    

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);
    
    Mat res;
    imgFrame2.copyTo(res);
    
    int intVerticalLinePosition = (int)round((double)imgFrame1.cols * 0.50);   //adjust it Y
    int intHorizontalLinePosition = (int)round((double)imgFrame1.rows * 0.50); //adjust it X
    
    
        //TODO Line for x axis first distance measure
  
    int intFirstLane = (int)round((double)imgFrame1.rows *0.20);
    int intSecondLane = (int)round((double)imgFrame1.rows * 0.65); //ito yung sucat sa second lane
//    int intHorizontalLinePositionLeft = (int)round((double)imgFrame1.rows * 0.45);  //unComment this line of code if you want to have a separeted line in left lane road.
    
    //Lines Properties point to end point

    //changes
    //Vertical y Line
    crossingLineDemo[1].x = intVerticalLinePosition;
    crossingLineDemo[1].y = 0;

    crossingLineDemo[0].x = intVerticalLinePosition;
    crossingLineDemo[0].y = imgFrame1.rows -1;
    
    
    //Horizontal x line
    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols -1;
    crossingLine[1].y = intHorizontalLinePosition;
    
    
     //TODO Line for x axis first distance measure
    dFirstLine[1].x = 250;
    dFirstLine[1].y = intFirstLane;

    dFirstLine[0].x = imgFrame1.rows -60;
    dFirstLine[0].y = intFirstLane;
    
    
    
    
    
   //TODO Line for x axis second distance measure
    dSecondLine[0].x = 200;
    dSecondLine[0].y = intSecondLane;

    dSecondLine[1].x = imgFrame1.rows -1;
    dSecondLine[1].y = intSecondLane;
    
    
    
    char chCheckForEscKey = 0;
    
    bool found = false;
    bool blnFirstFrame = true;
    
    int frameCount = 2;
    int notFound = 0;
    
    
    double frameCounter = 0;
    double ticks = 0;
   
    int tick = 0;
    int fps;
    std::time_t timeBegin = std::time(0);
    
    
    while (capVideo.isOpened() && chCheckForEscKey != 27) {
        
        //TODO GET the fps of the video
        frameCounter++;
        time_t timeNow = std::time(0) - timeBegin;
        if (timeNow - tick >= 1){
            tick++;
            fps = frameCounter;
            frameCounter = 0;
        }
      
        //get the seconds timelapse
        double preTick  = ticks ;
        double dt;
        
        ticks = (double)getTickCount();
        dt = (ticks - preTick) / getTickFrequency();
        
        
        //test print of current secconds
        
        cv::putText(imgFrame2, cv::format("Average fps=%d:%d",fps,frameCounter), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, SCALAR_YELLOW);

        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        if(found){
            // >>>> Matrix A
            kf.transitionMatrix.at<float>(2) = dt;
            kf.transitionMatrix.at<float>(9) = dt;
            // <<<< Matrix A
            
            cout<<"dt:"<<endl<<dt<<endl;
            state = kf.predict();
            
            cout<<"State post:"<<endl<<state<<endl;
            
            Rect currentBoundingRect;
            currentBoundingRect.width = state.at<float>(4);
            currentBoundingRect.height = state.at<float>(5);
            currentBoundingRect.x = state.at<float>(0) - currentBoundingRect.width / 2;
            currentBoundingRect.y = state.at<float>(1) - currentBoundingRect.height / 2;

            Point centerPositions;
            centerPositions.x = state.at<float>(0);
            centerPositions.y = state.at<float>(1);
            
            circle(res, centerPositions, 2, CV_RGB(255,0,0), -1);
            rectangle(res, currentBoundingRect, CV_RGB(255,0,0), 2);
            
        }
        
        
        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);
        
        //Blur Converting
        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, Size(5, 5), 3.3, 3.3);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, Size(5, 5), 3.3, 3.3);
        
        
        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30.30, 255.0, CV_THRESH_BINARY);
//        cv::imshow("imgThresh", imgThresh);

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3),Point(1,1));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5),Point(2,2));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7),Point(4,4));
        
        for (unsigned int i = 0; i < 2; i++) {                              //loping of the blob strcuture size .
            for(unsigned int a = 0; a < 1 ; a++){
                cv::dilate(imgThresh, imgThresh, structuringElement5x5);
                cv::dilate(imgThresh, imgThresh, structuringElement5x5);
                cv::erode(imgThresh, imgThresh, structuringElement3x3);    
            }
        }
        
        cv::Mat imgThreshCopy1 = imgThresh.clone();
        std::vector<std::vector<cv::Point> > contours;
        cv::findContours(imgThreshCopy1, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
//        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        
        
        //Filtering 
        vector<vector<Point> > convexHulls(contours.size());
        vector<vector<Point> > objectBlobs;
        vector<Rect> rectBox;
        
        
        
        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
            Rect rBox;
            rBox = boundingRect(contours[i]);
               float ratio = (float) rBox.width / (float) rBox.height;
            if (ratio > 1.0f)
                ratio = 1.0f / ratio;

            // Searching for a bBox almost square
            if (ratio > 0.75 && rBox.area() >= 400)
            {
                objectBlobs.push_back(contours[i]);
                rectBox.push_back(rBox);
            }
            std::cout << "Object found:" << rBox.size() << std::endl;
        }
        
        for (size_t i = 0; i < objectBlobs.size(); i++)
        {
            drawContours(imgFrame2Copy, objectBlobs, i, CV_RGB(20,150,20), 1);
            rectangle(imgFrame2Copy, rectBox[i], CV_RGB(0,255,0), 2);

            Point center;
            center.x = rectBox[i].x + rectBox[i].width / 2;
            center.y = rectBox[i].y + rectBox[i].height / 2;
            circle(imgFrame2Copy, center, 3, CV_RGB(20,150,20), -1);

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            putText(imgFrame2Copy, sstr.str(),Point(center.x + 3, center.y - 3),FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        }        
        

//        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");
        
        
        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);
            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.3 &&
                possibleBlob.dblCurrentAspectRatio < 5.0 &&
                possibleBlob.currentBoundingRect.width > 40 &&
                possibleBlob.currentBoundingRect.height > 40 &&
                possibleBlob.dblCurrentDiagonalSize > 3.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.3456) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

        if (blnFirstFrame == true) {
            for (auto currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs, imgFrame2Copy);
        }

//        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        drawBlobInfoOnImage(blobs, imgFrame2Copy,currentFrameBlobs,frameCounter, fps);
        
        
        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition,intVerticalLinePosition,carCount);
        bool carOnFirstLanePoint = carOnFirstLane(blobs, intFirstLane);
        bool carOnSecondLanePoint = carOnSecondLane(blobs, intFirstLane);
        
        
        //if object blob cross the line
        if (blnAtLeastOneBlobCrossedTheLine == true) {
            picNum++;
            //drawCarCountOnImage(carCount, carCountLeft, carCountNoLeft,imgFrame2Copy);
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 1);
            cv::line(imgFrame2Copy, crossingLineDemo[1], crossingLineDemo[0], SCALAR_RED, 2);
            cv::imwrite("C:/Users/Jovanie/Documents/NetBeansProjects/TrafficDemo/Data_Write/PIC"+intToString(picNum)+".tif",imgFrame2Copy);
            
        } else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
            cv::line(imgFrame2Copy, crossingLineDemo[1], crossingLineDemo[0], SCALAR_YELLOW, 1);
            
        }
        
        
        //TODO if(the car pass) through the first lane put the condition
        if(carOnFirstLanePoint == true){
             line(imgFrame2Copy, dFirstLine[0], dFirstLine[1],SCALAR_BLACK,0);
             timer1.start();
        }else{
            /**some condition here**/
            cv::line(imgFrame2Copy, dFirstLine[0], dFirstLine[1],SCALAR_WHITE,1);
            timer1.stop();
        }
        
        
        //TODO if(the car pass) the car pass through the second lane put the condition
        if(carOnSecondLanePoint == true){
            line(imgFrame2Copy, dSecondLine[0], dSecondLine[1],SCALAR_BLACK,0);
            timer2.start();
        }else{
            line(imgFrame2Copy, dSecondLine[0], dSecondLine[1],SCALAR_WHITE,1);
            timer2.stop();
        }
        
        drawCarCountOnImage(carCount, imgFrame2Copy);
//        imshow("imgFrame2Copy", imgFrame2Copy);
        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {
            capVideo.read(imgFrame2);
        } else {
            std::cout << "end of video\n";
            break;
        }
        
        if(objectBlobs.size()== 0){
            notFound++;
            cout<<"not found: "<< notFound<< endl;
             if( notFound >= 100 )
            {
                found = false;
            }
            
        }else{
        
            notFound = 0;

            meas.at<float>(0) = rectBox[0].x + rectBox[0].width / 2;
            meas.at<float>(1) = rectBox[0].y + rectBox[0].height / 2;
            meas.at<float>(2) = (float)rectBox[0].width;
            meas.at<float>(3) = (float)rectBox[0].height;

            if (!found){
                // >>>> Initialization
                kf.errorCovPre.at<float>(0) = 1;
                kf.errorCovPre.at<float>(7) = 1;
                kf.errorCovPre.at<float>(14) = 1;
                kf.errorCovPre.at<float>(21) = 1;
                kf.errorCovPre.at<float>(28) = 1;
                kf.errorCovPre.at<float>(35) = 1;

                state.at<float>(0) = meas.at<float>(0);
                state.at<float>(1) = meas.at<float>(1);
                state.at<float>(2) = 0;
                state.at<float>(3) = 0;
                state.at<float>(4) = meas.at<float>(2);
                state.at<float>(5) = meas.at<float>(3);
                // <<<< Initialization

                kf.statePost = state;
                found = true;

            }
            else
                kf.correct(meas); // Kalman Correction
                cout << "Measure matrix:" << std::endl << meas << std::endl;
        }
        
        for(size_t i = 0 ; i < objectBlobs.size(); i++){
           drawContours(imgFrame2Copy, objectBlobs, i, CV_RGB(20,150,20), 1);
           rectangle(imgFrame2Copy, rectBox[i], CV_RGB(0,255,0), 2);

            Point center;
            center.x = rectBox[i].x + rectBox[i].width / 2;
            center.y = rectBox[i].y + rectBox[i].height / 2;
            circle(imgFrame2Copy, center, 3, CV_RGB(20,150,20), -1); //drawing a dot circle in the tracking object

            stringstream sstr;
            sstr << "(" << center.x << "," << center.y << ")";
            putText(imgFrame2Copy, sstr.str(),Point(center.x + 3, center.y - 3),FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(20,150,20), 2);
        }
        
        imshow("FrameVideoCam",imgFrame2Copy);
        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = waitKey(1);
    }

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }

    return(0);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs, cv::Mat &imgFrame2Copy) {

    for (auto &existingBlob : existingBlobs) {
        existingBlob.blnCurrentMatchFoundOrNewBlob = false;
        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0;
      
        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);
              
                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
//                    std::cout<<"This is the leastDistance:"<<std::to_string(dblLeastDistance)+"\n";
//                    cv::putText(imgFrame2Copy,std::to_string(dblDistance), existingBlobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
                }
            }
        }

        
        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto &existingBlob : existingBlobs) {

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);
    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);
    cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {//

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &intVerticalLinePosition, int &carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {

        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y > intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y <= intHorizontalLinePosition || blob.centerPositions[prevFrameIndex].y < intVerticalLinePosition&& blob.centerPositions[currFrameIndex].y >= intVerticalLinePosition) {
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true; 
                
            }
//            std::cout<<"This is the Cureent FrameIndex"<<std::to_string(currFrameIndex);
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//TODO bool in first lane.
bool carOnFirstLane(vector<Blob> &blobs, int &intFirstLane){
   bool blbOnFirstLane = false;
   
   for(auto blob: blobs){
      
       if(blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2){
           int prevFrameIndex = (int)blob.centerPositions.size() - 2;
           int currFrameIndex = (int)blob.centerPositions.size() - 1;
           if(blob.centerPositions[prevFrameIndex].y < intFirstLane && blob.centerPositions[currFrameIndex].y >= intFirstLane){
                blbOnFirstLane = true;
                cout<<"dumaan sa first lane"<<"\n";
           }
       }
   }

   return blbOnFirstLane;
}
//TODO bool in second lane.
bool carOnSecondLane(vector<Blob> &blobs, int &intSecondLane){
    bool blbOnSecondLane = false;
    for(auto blob : blobs){
        if(blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2){
            int prevFrameIndex = (int)blob.centerPositions.size() -2 ;
            int currFrameIndex = (int)blob.centerPositions.size() -1 ;
            if(blob.centerPositions[prevFrameIndex].y < intSecondLane && blob.centerPositions[currFrameIndex].y >= intSecondLane){
                blbOnSecondLane = true;
                cout<<"dumaan sa second lane"<<"\n";
            }
        }
    }
    return blbOnSecondLane;
}




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy,std::vector<Blob> &currentFrameBlobs,double &frameCounter, int &fps) {
    
    for (auto &currentFrameBlob : currentFrameBlobs){
    
        int indexLeastDistance = 0;
        double leastDistance = 10000.0;
        double time;
        time = frameCounter/fps; 
        
        for (unsigned int i = 0; i < blobs.size(); i++) {
            
               int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
                double dblFontScale = blobs[i].dblCurrentDiagonalSize / 350.0;
                int intFontThickness = (int)std::round(dblFontScale * 1.0);
//                cv::circle(imgFrame2Copy, center, 2, CV_RGB(20,150,20), 1);
//                string lastData;
                
            
        //        double distance = distanceBetweenPoints(blobs[]);
            if (blobs[i].blnStillBeingTracked == true && blobs[i].centerPositions.size() >= 2) {
                int prevFrameIndex = (int)blobs[i].centerPositions.size() - 2;
                int currFrameIndex = (int)blobs[i].centerPositions.size() - 1;
                double distance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(),blobs[i].predictedNextPosition);
                string data;
                int y,x;
                y = prevFrameIndex;
                x = currFrameIndex;               
                
                if (distance < leastDistance) {
                    leastDistance = distance;
                    indexLeastDistance = i;
//                    std::cout<<"This is the leastDistance:"<<std::to_string(distance)+"\n";
                  
                }
                
                double speed = leastDistance/time;
                double speedKph = speed*1000/3600;
                data = std::to_string(speedKph);    
                  

//                cv::Point center;
//                cout<<data+"\n";
//                rectangle(imageF);
              //cv::circle(imgFrame2Copy,blobs[i].currentBoundingRect,SCALAR_BLACK, 1);
//                cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_GREEN, 2);
//                cv::putText(imgFrame2Copy,format("(Y:%d,X:%d)",y,x), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

            }
        }
    
    }
}


//void drawBlobInfoImage()
///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {
    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1.5);

    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

    cv::Point ptTextBottomLeftPosition;

    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);
    
    cv::rectangle(imgFrame2Copy,cv::Rect(0,400,200,20),SCALAR_WHITE,-1);
    cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
    cv::putText(imgFrame2Copy,getDateTime(), cv::Point(0,410), cv::FONT_HERSHEY_SIMPLEX,0.4, SCALAR_BLACK ,1.5);
}


