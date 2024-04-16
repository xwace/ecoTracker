#pragma once
#include <opencv2/opencv.hpp>
namespace eco
{
	void init_eco(cv::Mat &img, cv::Rect2f &rect);
	bool update_eco(cv::Mat &img, cv::Rect2f &rect, float score);
}
