#pragma once

#include "parameters.h"
#include "scale_filter.h"
#include "feature_extractor.h"
#include "sample_update.h"
#include "training.h"
#include<core/types_c.h>

namespace eco
{
	class ECO
	{
	public:
		ECO() {};
		virtual ~ECO() {}

		void init(cv::Mat &im, const cv::Rect2f &rect);

		bool update(const cv::Mat &frame, cv::Rect2f &roi, float score);

		void init_parameters(const eco::EcoParameters &parameters);

		void init_features();

		void yf_gaussian(); // the desired outputs of features, real part of (9) in paper C-COT  特征的期望输出，C-COT论文公式9的实数部分

		void cos_window(); 	// construct cosine window of features; 构造余弦特征的窗口

		ECO_FEATS interpolate_dft(const ECO_FEATS &xlf,
			std::vector<cv::Mat> &interp1_fs,
			std::vector<cv::Mat> &interp2_fs);

		ECO_FEATS compact_fourier_coeff(const ECO_FEATS &xf);

		ECO_FEATS full_fourier_coeff(const ECO_FEATS &xf);

		std::vector<cv::Mat> project_mat_energy(std::vector<cv::Mat> proj,
			std::vector<cv::Mat> yf);

		ECO_FEATS shift_sample(ECO_FEATS &xf,
			cv::Point2f shift,
			std::vector<cv::Mat> kx,
			std::vector<cv::Mat> ky);

	private:
		bool				is_color_image_;
		EcoParameters 		params_;
		cv::Point2f 		pos_; 							// final result  最终结果
		size_t 				frames_since_last_train_; 	 	// used for update;  用于更新

		// The max size of feature and its index, output_sz is T in (9) of C-COT paper  output_size_ 是 C-COT论文公式9中的T， output_size_是特征的最大尺寸，以及它的索引
		size_t 				output_size_, output_index_;

		cv::Size2f 			base_target_size_; 	// target size without scale  目标尺寸，而不是比例
		cv::Size2i			img_sample_size_;  	// base_target_sz * sarch_area_scale 图片中样本图片的大小 sqrt(base_target_size_.area() * std::pow(params_.search_area_scale, 2))
		cv::Size2i			img_support_size_;	// the corresponding size in the image  图片中样本图片的大小

		std::vector<cv::Size> 	feature_size_, filter_size_;
		std::vector<int> 		feature_dim_, compressed_dim_;

		ScaleFilter 		scale_filter_;
		int 				nScales_;				// number of scales;
		float 				scale_step_;
		std::vector<float>		scale_factors_;
		float 				currentScaleFactor_; 	// current img scale 当前图片的缩放比例因子

		// Compute the Fourier series indices 计算傅立叶级数指数
		// kx_, ky_ is the k in (9) of C-COT paper, yf_ is the left part of (9); kx_, ky_ C-COT论文公式9中的实部和虚部， yf_ 公式9的等号左边部分
		std::vector<cv::Mat> 	ky_, kx_, yf_;
		std::vector<cv::Mat> 	interp1_fs_, interp2_fs_;
		std::vector<cv::Mat> 	cos_window_;
		std::vector<cv::Mat> 	projection_matrix_;

		std::vector<cv::Mat> 	reg_filter_;
		std::vector<float> 		reg_energy_;

		FeatureExtractor 	feature_extractor_;

		SampleUpdate 		sample_update_;
		ECO_FEATS 			sample_energy_;

		EcoTrain 			eco_trainer_;

		ECO_FEATS 			hf_full_;
	};
}
