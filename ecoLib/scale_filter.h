#pragma once
#include <opencv2/opencv.hpp>
#include "parameters.h"
namespace eco
{
	class ScaleFilter
	{
	public:
		ScaleFilter() {};
		virtual ~ScaleFilter() {};
		void init(int &nScales, float &scale_step, const EcoParameters &params);
		float scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz, const float &currentScaleFactor, const EcoParameters &params);
		cv::Mat extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz, std::vector<float> &scaleFactors, const cv::Size &scale_model_sz);

	private:
		std::vector<float> scaleSizeFactors_;
		std::vector<float> interpScaleFactors_;
		cv::Mat yf_;
		std::vector<float> window_;
		bool max_scale_dim_;
	};
} // namespace eco
