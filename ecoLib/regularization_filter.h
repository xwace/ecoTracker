#pragma once

#include <opencv2/opencv.hpp>
#include "parameters.h"
namespace eco
{
	cv::Mat get_regularization_filter(cv::Size sz,
		cv::Size2f target_sz,
		const EcoParameters &params);
}
