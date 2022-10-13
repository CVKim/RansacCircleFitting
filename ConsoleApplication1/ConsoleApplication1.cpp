// ctr_circlefitting.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//



#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <math.h>
#include <cmath>

using namespace std;
using namespace cv;

// 전방 선언
double calculateDistance(std::pair<int, int>& x, std::pair<int, int>& y);
double calculateDistance(std::pair<double, double>& x, std::pair<double, double>& y);
double calculateDistance(cv::Point2f temp1, cv::Point2f temp2);
Point2f ScalePoint(cv::Point2f center, cv::Point2f pt, float scale);
double getRadian(int _num);
void cvImgShow(string winName, Mat& mat);
void timerCout(time_t& end, time_t& start);

#define _USE_MATH_DEFINES
#define PI 3.1415926535897
RNG rng(12345);

class Points
{
public:
    float x = 0;
    float y = 0;

public:
    Points(float x, float y) { this->x = x + 0.5, this->y = y + 0.5; }
};

double calculateDistance(std::pair<int, int>& x, std::pair<int, int>& y)
{
    return sqrt(pow(x.first - y.first, 2) +
        pow(x.second - y.second, 2));
}
double calculateDistance(std::pair<double, double>& x, std::pair<double, double>& y)
{
    return sqrt(pow(x.first - y.first, 2) +
        pow(x.second - y.second, 2));
}
double calculateDistance(cv::Point2f temp1, cv::Point2f temp2)
{
    return sqrt(pow(temp2.x - temp1.x, 2) + pow(temp2.y - temp1.y, 2));
    /* return sqrt(pow(x.first - y.first, 2) +
         pow(x.second - y.second, 2));*/
}

Point2f ScalePoint(cv::Point2f center, cv::Point2f pt, float scale)
{
    int delta_x, delta_y;
    delta_x = center.x - pt.x;
    delta_y = center.y - pt.y;

    float x_scale, y_scale;
    x_scale = center.x - scale * delta_x;
    y_scale = center.y - scale * delta_y;

    return cv::Point2f(x_scale + 0.5, y_scale + 0.5);
}
double getRadian(int _num)
{
    return _num * (PI / 180);
}
void cvImgShow(string winName, Mat& mat)
{
    cv::imshow(winName, mat);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
void timerCout(time_t& end, time_t& start)
{
    double resultTime;

    end = time(NULL);
    resultTime = (double)(end - start);
    std::printf("result time : %d", resultTime);
}

int main()
{

    bool isEven = true;

    time_t start, end;
    start = time(NULL);
    Mat src, gray, gaussian, outMat;

    // 코깅 편심 검사 할 때, 사용되는 filter
    // unsharp mask filter

    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;

    vector<vector<cv::Point2f>> points_c1(3);
    vector<vector<cv::Point2f>> points_c2(3);

    vector<cv::Point2f> points_inner;
    vector<cv::Point2f> points_outer;

    vector<cv::Point2f> points_join;
    vector<cv::Point2f> points_det;

    vector<float> ratios;
    vector<float> ratios_long;

    vector<float> pointsMin;
    vector<float> pointsMax;

    // circle inner ~ outer width val
    vector<float> cirWidthVal;


    src = imread("d:\\temp1.bmp");
    Mat ddd;
    src.convertTo(ddd, CV_32S);

    cvtColor(src, gray, cv::COLOR_RGB2GRAY);


    if (isEven) outMat = gray;
    else
    {
        auto alpha = 4;             // Intensity
        auto beta = -1 * (3 + 0.5); // Intensity + Gamma
        cv::GaussianBlur(gray, gaussian, cv::Size(0.1, 0.1), 5);
        cv::addWeighted(gray, alpha, gaussian, beta, 0, outMat);
    }

    cv::findContours(outMat, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);



    vector<vector<cv::Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    int ii = 0;
    int checkNum = contours.size();
    vector<vector<cv::Point>> tempVec(6);

    for (int i = 0; i < checkNum; i++) {

        int area = contourArea(contours[i]);

        if (area < 70000) continue;

        tempVec[ii].assign(contours[i].begin(), contours[i].end());
        ii++;
        float peri = arcLength(contours[i], true);
        bool bClosed = true;
        approxPolyDP(contours[i], conPoly[i], 0.01 * peri, bClosed);
        boundRect[i] = boundingRect(conPoly[i]);
        Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);

        std::cout << "개구부 Width : " << std::fabs((boundRect[i].tl().x - boundRect[i].br().x) * 39.5 / 1000) << std::endl;

        /*rectangle(src, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 255), 1);*/
        // 꼭지점 갯수
        int vtc = conPoly[i].size();

        printf("Number %i,   conpoly count : %i", i, vtc);

        // 장-단축 반지름 구하기 위함
        RotatedRect ellipse = cv::fitEllipse(contours[i]);

        cv::Point2f cenXY;
        float radius;
        double ratio;
        cv::minEnclosingCircle(contours[i], cenXY, radius);
        ratio = radius * radius * PI / area;

        ratios.push_back(ceil(ellipse.size.width / 2));
        ratios_long.push_back(ellipse.size.height / 2);

        cv::Point2i center{ int(cenXY.x), int(cenXY.y) };

        cv::circle(src, center, ratio, cv::Scalar(0, 255, 0), 1.5);
        cv::drawContours(src, contours, i, cv::Scalar(0, 0, 255), 1);

        cv::putText(src, to_string(i), cv::Point(boundRect[i].x, boundRect[i].y), cv::FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1);

        points_det.push_back(cv::Point2f(cenXY.x, cenXY.y));
        cout << "노드 좌표 : " << cenXY.x << ' ' << cenXY.y << '\n';
    }

    // --------------------------------------------------
    int x1, x2;
    x1 = 0;
    x2 = 1;

    double dist, cosRadian, sinRadian;
    float dist_remain;

    vector<int> bufCnt;
    vector<std::map<string, cv::Point>> mapStEnPos(36);
    if (isEven) {

        for (int angle = 0; angle < 360; angle += 10) {

            float x_1, x_2, y_1, y_2;

            int temp = points_det.size();

            x_1 = ratios_long[x1] * cos(getRadian(angle)) + points_det[x1].x;
            y_1 = ratios_long[x1] * sin(getRadian(angle)) + points_det[x1].y;
            points_inner.push_back(cv::Point2f(x_1, y_1));

            x_2 = ratios_long[x2] * cos(getRadian(angle)) + points_det[x2].x;
            y_2 = ratios_long[x2] * sin(getRadian(angle)) + points_det[x2].y;
            points_outer.push_back(cv::Point2f(x_2, y_2));

            cout << "x_2 : " << x_2 << ", y_2 : " << y_2 << " end" << endl;
        }

        if (tempVec[x1].capacity() == 0) return 0;


        // key val 중복 될 수도 있으므로
        // 이중 vector 구현 필요
        std::vector<cv::Point> vecTmpPos;


        int tmpNum;
        for (int angle = 0; angle < 36; angle += 1) {

            /*auto it11 = find(points_det.begin(), points_det.end(), 0);*/
            LineIterator it(gray, points_outer[angle], points_det[x1], 8);
            vector<uchar> buffer(it.count);

            std::cout << "Line interator Dis : " << it.count << std::endl;

            for (int ii = 0; ii < it.count; ii++, ++it) {
                buffer[ii] = **it;
                vecTmpPos.push_back(it.pos()); // 좌표
            }

            //for (std::vector<uchar>::iterator it = buffer.begin(); it != buffer.end(); ++it)
            //{
            //    std::cout << (int)*it << "buffer out" << endl;
            //}

            bufCnt.push_back(count_if(buffer.begin(), buffer.end(), [](int num) {return num >= 1; }));


            // 이중 vector 필요
            bool b_reverse = false;
            for (int i = 0; i < buffer.size(); i++) {
                auto it = buffer[i];
                if ((int)it == 255 && !b_reverse) {
                    mapStEnPos[angle].insert(std::pair("start", (vecTmpPos[i])));
                    mapStEnPos[angle].insert(std::pair("end", (vecTmpPos[i + bufCnt[angle] - 1])));
                    b_reverse = true; // 255가 아닌 0을 찾을 때 필요 할 수도??
                    break;
                }
            }
            vecTmpPos.clear();

            dist = calculateDistance(points_outer[angle], points_det[x1]);
            dist_remain = dist - ratios_long[x1];
        }
    }
    else {

        x1 = 1;
        x2 = 3;

        for (int angle = 0; angle < 360; angle += 10) {

            float x_1, x_2, y_1, y_2;

            x_1 = ratios_long[x1] * cos(getRadian(angle)) + points_det[x1].x;
            y_1 = ratios_long[x1] * sin(getRadian(angle)) + points_det[x1].y;
            points_inner.push_back(cv::Point2f(x_1, y_1));

            x_2 = ratios_long[x2] * cos(getRadian(angle)) + points_det[x2].x;
            y_2 = ratios_long[x2] * sin(getRadian(angle)) + points_det[x2].y;
            cout << "x_2 : " << x_2 << ", y_2 : " << y_2 << " end" << endl;

            points_outer.push_back(cv::Point2f(x_2, y_2));
        }

        for (int angle = 0; angle < 36; angle++) {
            cv::line(src, points_outer[angle], points_det[x1], Scalar(0, 0, 255), 1, 8, 0);
        }

        for (int angle = 0; angle < 36; angle++) {

            // 외곽 pixel pos(x,y) - 내곽 circle center(x,y)
            double dist = calculateDistance(points_outer[angle], points_det[x1]);
            float dist_remain = dist - ratios_long[x1];

            points_join.push_back(ScalePoint(points_outer[angle], points_det[x1], dist_remain / dist));

            cirWidthVal.push_back(dist_remain);

        }

        x1 = 0;
    }

    float   min, max;
    int     minIdx, maxIdx;

    if (isEven) {

        auto it = minmax_element(bufCnt.begin(), bufCnt.end());

        min = *it.first;
        max = *it.second;

        minIdx = distance(bufCnt.begin(), it.first);
        maxIdx = distance(bufCnt.begin(), it.second);

        auto minPos = mapStEnPos.at(minIdx);
        auto maxPos = mapStEnPos.at(maxIdx);


        rectangle(src, cv::Point(minPos["start"].x, minPos["start"].y), cv::Point(minPos["end"].x, minPos["end"].y), Scalar(0, 255, 0), 2, 8, 0);
        rectangle(src, cv::Point(maxPos["start"].x, maxPos["start"].y), cv::Point(maxPos["end"].x, maxPos["end"].y), Scalar(100, 100, 255), 2, 8, 0);

        //float outToInX = std::fabs(points_inner[minIdx].x - points_outer[minIdx].x);
        //float outToInY = std::fabs(points_inner[minIdx].y - points_outer[minIdx].y);
        //
        //float drawMinXPos = (points_inner[minIdx].x < points_outer[minIdx].x) ? points_inner[minIdx].x : points_outer[minIdx].x;
        //float drawMaxXPos = (points_inner[maxIdx].x < points_outer[maxIdx].x) ? points_inner[maxIdx].x : points_outer[maxIdx].x;
        //
        //float drawMinYpos = (points_inner[minIdx].y < points_outer[minIdx].y) ? points_inner[minIdx].y : points_outer[minIdx].y;
        //float drawMaxYpos = (points_inner[maxIdx].y < points_outer[maxIdx].y) ? points_inner[maxIdx].y : points_outer[maxIdx].y;
        //
        //
        //rectangle(src, Rect2f(drawMinXPos - 5, drawMinYpos, 40, 40), Scalar(0, 255, 0), 2, 8, 0);
        //rectangle(src, Rect2f(drawMaxXPos - 5, drawMaxYpos, 40, 40), Scalar(100, 100, 255), 2, 8, 0);

    }
    else {

        auto it = minmax_element(cirWidthVal.begin(), cirWidthVal.end());

        min = *it.first;
        max = *it.second;

        minIdx = distance(cirWidthVal.begin(), it.first);
        maxIdx = distance(cirWidthVal.begin(), it.second);

        rectangle(src, Rect2f(points_join[minIdx].x - 25, points_join[minIdx].y - 25, 55, 50), Scalar(100, 255, 100), 3, 8, 0);
        rectangle(src, Rect2f(points_join[maxIdx].x - 25, points_join[maxIdx].y - 25, 55, 50), Scalar(100, 100, 255), 3, 8, 0);
    }

    /*imwrite("d:\\temp888_out.bmp", src);*/

    cirWidthVal.clear();
    points_join.clear();
    points_outer.clear();
    points_inner.clear();

    for (auto const& item : points_det)
        cout << "center x : " << item.x << ",  center y : " << item.y << endl;

    for (auto const& item : ratios_long)
        cout << "ratios : " << item << endl;

    timerCout(end, start);
    cvImgShow("OutMat Image", src);
    cv::imwrite("d:\\temps.bmp", src);
    return 0;
}