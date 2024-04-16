// ecoLib.cpp : 定义静态库的函数。
//
#include "eco_lib.h"
#include "eco.h"

namespace eco
{
	ECO ecotracker;

	void init_eco(cv::Mat &img, cv::Rect2f &rect)
	{
		ecotracker.init(img, rect);
	}

	bool update_eco(cv::Mat &img, cv::Rect2f &rect, float score)
	{
		return ecotracker.update(img, rect, score);
	}
}

