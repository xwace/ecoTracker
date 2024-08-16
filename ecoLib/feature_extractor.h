#pragma once
#include "parameters.h"
namespace eco
{
	class FeatureExtractor
	{
	public:
		FeatureExtractor() {}
		virtual ~FeatureExtractor() {};

		ECO_FEATS extractor(const cv::Mat image,
			const cv::Point2f pos,
			const std::vector<float> scales,
			const EcoParameters &params,
			const bool &is_color_image);

		cv::Mat sample_patch(const cv::Mat im,
			const cv::Point2f pos,
			cv::Size2f sample_sz,
			cv::Size2f input_sz);

		std::vector<cv::Mat> get_hog_features(const std::vector<cv::Mat> ims);
		std::vector<cv::Mat> hog_feature_normalization(std::vector<cv::Mat> &hog_feat_maps);
		inline std::vector<cv::Mat> get_hog_feats() const { return hog_feat_maps_; }

		std::vector<cv::Mat> get_cn_features(const std::vector<cv::Mat> ims);
		std::vector<cv::Mat> cn_feature_normalization(std::vector<cv::Mat> &cn_feat_maps);
		inline std::vector<cv::Mat> get_cn_feats() const { return cn_feat_maps_; }

	private:
		EcoParameters params_;

		HogFeatures hog_features_;
		int hog_feat_ind_ = -1;
		std::vector<cv::Mat> hog_feat_maps_;


		CnFeatures cn_features_;
		int cn_feat_ind_ = -1;
		std::vector<cv::Mat> cn_feat_maps_;
	};
}
