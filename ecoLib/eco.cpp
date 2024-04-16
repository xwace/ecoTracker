#include "eco.h"
#include "interpolator.h"
#include "regularization_filter.h"
#include "ffttools.h"
#include <vector>
#include "recttools.h"
#include "optimize_scores.h"
#include <numeric>
#include <fstream>
#include "cn_norm.h"

namespace eco
{
	void ECO::init(cv::Mat &im, const cv::Rect2f &rect)
	{
		//debug("=========================Init================================");
		double timereco = (double)cv::getTickCount();
		float fpseco = 0;
		// 0. Clean all the parameters.
		pos_.x = 0;
		pos_.y = 0;
		frames_since_last_train_ = 0;
		output_size_ = 0;
		output_index_ = 0;
		base_target_size_.height = 0;
		base_target_size_.width = 0;
		img_sample_size_.height = 0;
		img_sample_size_.width = 0;
		img_support_size_.height = 0;
		img_support_size_.width = 0;
		feature_size_.clear();
		filter_size_.clear();
		feature_dim_.clear();
		compressed_dim_.clear();
		currentScaleFactor_ = 0;
		nScales_ = 0;
		ky_.clear();
		kx_.clear();
		yf_.clear();
		cos_window_.clear();
		interp1_fs_.clear();
		interp2_fs_.clear();
		reg_filter_.clear();
		projection_matrix_.clear();
		reg_energy_.clear();
		scale_factors_.clear();
		sample_energy_.clear();
		hf_full_.clear();

		// 
		cv::ipp::setUseIPP(false);

		// 1. Initialize all the parameters.
		/*
	*/
	// Close opencl to cut the init time
		//cv::ocl::setUseOpenCL(false);

		// Image infomations
		if (im.channels() == 3)
		{
			is_color_image_ = true;
		}
		else
		{
			is_color_image_ = false;
		}
		//	printMat(im);
			//debug("is_color_image_: %d, rect: %f, %f, %f, %f", is_color_image_, rect.x, rect.y, rect.width, rect.height);

			// Get the ini position 中心点位置
		pos_.x = rect.x + (rect.width - 1.0) / 2.0;
		pos_.y = rect.y + (rect.height - 1.0) / 2.0;
		//debug("pos_:%f, %f", pos_.y, pos_.x);

		EcoParameters paramters;
		// Read in all the parameters
		init_parameters(paramters);

		// Calculate search area and initial scale factor ，搜索面积在原有的面积基础上变为16倍
		float search_area = rect.area() * std::pow(params_.search_area_scale, 2);
		//debug("search_area:%f", search_area);
		if (search_area > params_.max_image_sample_size)
			currentScaleFactor_ = sqrt((float)search_area / params_.max_image_sample_size);
		else if (search_area < params_.min_image_sample_size)
			currentScaleFactor_ = sqrt((float)search_area / params_.min_image_sample_size);
		else
			currentScaleFactor_ = 1.0;
		currentScaleFactor_ = 1.0;
		//debug("currentscale:%f", currentScaleFactor_);

		// target size at the initial scale
		base_target_size_ = cv::Size2f(rect.size().width / currentScaleFactor_, rect.size().height / currentScaleFactor_);
		//debug("base_target_size_:%f x %f", base_target_size_.height, base_target_size_.width);

		// window size, taking padding into account
		if (params_.search_area_shape == "proportional")
		{
			img_sample_size_ = base_target_size_ * params_.search_area_scale;
		}
		else if (params_.search_area_shape == "square")
		{
			float img_sample_size__tmp;
			img_sample_size__tmp = sqrt(base_target_size_.area() * std::pow(params_.search_area_scale, 2));
			img_sample_size_ = cv::Size2i(img_sample_size__tmp, img_sample_size__tmp);
		}
		else if (params_.search_area_shape == "fix_padding")
		{
		}
		else if (params_.search_area_shape == "custom")
		{
		}
		//debug("img_sample_size_: %d x %d", img_sample_size_.height, img_sample_size_.width);

		init_features();

		// Number of Fourier coefficients to save for each filter layer. This will be an odd number. 为每个滤波器层保存的傅立叶系数个数。这将是一个奇数。
		output_index_ = 0;
		output_size_ = 0;
		// The size of the label function DFT. Equal to the maximum filter size  标签函数DFT的大小。等于最大过滤器大小
		for (size_t i = 0; i != feature_size_.size(); ++i)  // feature_size_ 保存使用到的特征（CNN、HOG、CN、IC）每个特征的大小
		{
			size_t size = feature_size_[i].width + (feature_size_[i].width + 1) % 2; //=63, to make it as an odd number;  变成一个基数
			filter_size_.push_back(cv::Size(size, size));

			// get the largest feature and it's index; 获取最大特征及其索引
			output_index_ = size > output_size_ ? i : output_index_;
			output_size_ = std::max(size, output_size_);
		}
		//debug("output_index_:%lu, output_size_:%lu", output_index_, output_size_);

		// Compute the 2d Fourier series indices by kx and ky.  通过kx和ky计算二维傅里叶级数指数
		for (size_t i = 0; i < filter_size_.size(); ++i) // for each filter
		{
			cv::Mat_<float> tempy(filter_size_[i].height, 1, CV_32FC1);
			cv::Mat_<float> tempx(1, filter_size_[i].height / 2 + 1, CV_32FC1);

			// ky in [-(N-1)/2, (N-1)/2], because N = filter_size_[i].height is odd (check above), N x 1;
			for (int j = 0; j < tempy.rows; j++)
			{
				tempy.at<float>(j, 0) = j - (tempy.rows / 2); // y index
			}
			ky_.push_back(tempy);
			// kx in [-N/2, 0], 1 x (N / 2 + 1)
			for (int j = 0; j < tempx.cols; j++)
			{
				tempx.at<float>(0, j) = j - (filter_size_[i].height / 2);
			}
			kx_.push_back(tempx);

		}
		// Construct the Gaussian label function using Poisson formula  利用泊松公式构造高斯标签函数
		yf_gaussian();  // 即计算C-COT论文公式9
		// Construct cosine window  构造余弦窗口
		cos_window();
		// Compute Fourier series of interpolation function, refer C-COT  计算插值函数的傅里叶级数，参见C-COT
		for (size_t i = 0; i < filter_size_.size(); ++i) // for each feature
		{
			cv::Mat interp1_fs1, interp2_fs1;
			Interpolator::get_interp_fourier(filter_size_[i],
				interp1_fs1,
				interp2_fs1,
				params_.interpolation_bicubic_a);
			interp1_fs_.push_back(interp1_fs1);
			interp2_fs_.push_back(interp2_fs1);
		}
		// Construct spatial regularization filter, refer SRDCF 构造空间正则化滤波器，参见SRDCF
		for (size_t i = 0; i < filter_size_.size(); i++) // for each feature
		{
			cv::Mat temp_d = get_regularization_filter(img_support_size_,
				base_target_size_,
				params_);
			cv::Mat temp_f;
			temp_d.convertTo(temp_f, CV_32FC1);
			reg_filter_.push_back(temp_f);
			

			// Compute the energy of the filter (used for preconditioner)	drone_flip
			cv::Mat_<double> t = temp_d.mul(temp_d); //element-wise multiply
			float energy = mat_sum_d(t);			 //sum up all the values of each points of the mat
			reg_energy_.push_back(energy);
			//debug("reg_energy_ %lu: %f", i, energy);
		}

		if (params_.use_scale_filter) // fDSST
		{
			scale_filter_.init(nScales_, scale_step_, params_);
			if (params_.scale_model_factor * params_.scale_model_factor * rect.area() > params_.scale_model_max_area)
			{
				params_.scale_model_factor = std::sqrt(params_.scale_model_max_area / rect.area());
			}

			params_.scale_model_sz.height = std::max((int)std::floor(rect.height * params_.scale_model_factor), 8);
			params_.scale_model_sz.width = std::max((int)std::floor(rect.width * params_.scale_model_factor), 8);

			params_.s_num_compressed_dim = nScales_;
			scale_factors_.push_back(1.0f);
		}
		else // SAMF
		{
			nScales_ = params_.number_of_scales;
			scale_step_ = params_.scale_step;
			if (nScales_ % 2 == 0)
			{
				nScales_++;
			}
			int scalemin = floor((1.0 - (float)nScales_) / 2.0);
			int scalemax = floor(((float)nScales_ - 1.0) / 2.0);
			for (int i = scalemin; i <= scalemax; i++)
			{
				scale_factors_.push_back(std::pow(scale_step_, i));
			}
		}
		if (nScales_ > 0)
		{
			params_.min_scale_factor = //0.01;
				std::pow(scale_step_,
					std::ceil(
						std::log(
							std::fmax(5 / (float)img_support_size_.width,
								5 / (float)img_support_size_.height)) /
						std::log(scale_step_)));
			params_.max_scale_factor = //10;
				std::pow(scale_step_,
					std::floor(
						std::log(
							std::fmin(im.cols / (float)base_target_size_.width,
								im.rows / (float)base_target_size_.height)) /
						std::log(scale_step_)));
		}
		//debug("scalefactor min: %f max: %f", params_.min_scale_factor, params_.max_scale_factor);

		// Set conjugate gradient options
		params_.CG_opts.CG_use_FR = true;
		params_.CG_opts.tol = 1e-6;
		params_.CG_opts.CG_standard_alpha = true;
		params_.CG_opts.debug = params_.debug;
		if (params_.CG_forgetting_rate == INF || params_.learning_rate >= 1)
		{
			params_.CG_opts.init_forget_factor = 0;
		}
		else
		{
			params_.CG_opts.init_forget_factor = std::pow(1.0f - params_.learning_rate, params_.CG_forgetting_rate);
		}
		//params_.CG_opts.init_forget_factor = 1;
		params_.CG_opts.maxit = std::ceil(params_.init_CG_iter / params_.init_GN_iter);
		//debug("-------------------------------------------------------------");
		ECO_FEATS xl, xlf, xlf_proj;

		// 2. Extract features from the first frame.
		xl = feature_extractor_.extractor(im, pos_, std::vector<float>(1, currentScaleFactor_), params_, is_color_image_);
		/*
		for (int i = 0; i < xl.size(); i++)
		{
			for (int j = 0; j < xl[i].size(); j++)
			{
				std::cout << i << ", " << j << std::endl;
				std::cout << xl[i][j] << std::endl;
			}
		}
		*/
		// 3. Multiply the features by the cosine window.
		xl = do_windows(xl, cos_window_);

		

		// 4. Do DFT on the features.
		xlf = do_dft(xl);

		

		// 5. Interpolate features to the continuous domain.
		xlf = interpolate_dft(xlf, interp1_fs_, interp2_fs_);

		xlf = compact_fourier_coeff(xlf); // take half of the cols

		

		// 6. Initialize projection matrix P.
		projection_matrix_ = init_projection_matrix(xl, compressed_dim_, feature_dim_); // 32FC2 31x10

		// 7. Do the feature reduction for each feature.
		xlf_proj = FeatureProjection(xlf, projection_matrix_);

		// 8. Initialize and update sample space.
		sample_update_.init(filter_size_, compressed_dim_, params_.nSamples, params_.learning_rate);
		sample_update_.update_sample_space_model(xlf_proj);

		// 9. Calculate sample energy and projection map energy.
		sample_energy_ = FeautreComputePower2(xlf_proj);
		std::vector<cv::Mat> proj_energy = project_mat_energy(projection_matrix_, yf_);

		// 10. Initialize filter and it's derivative.
		ECO_FEATS hf, hf_inc;
		for (size_t i = 0; i < xlf_proj.size(); i++) // for each feature
		{
			hf.push_back(std::vector<cv::Mat>(xlf_proj[i].size(), cv::Mat::zeros(xlf_proj[i][0].size(), CV_32FC2)));
			hf_inc.push_back(std::vector<cv::Mat>(xlf_proj[i].size(), cv::Mat::zeros(xlf_proj[i][0].size(), CV_32FC2)));
		}

		// 11. Train the tracker(train the filter and update the projection matrix).
		eco_trainer_.train_init(hf,
			hf_inc,
			projection_matrix_,
			xlf,
			yf_,
			reg_filter_,
			sample_energy_,
			reg_energy_,
			proj_energy,
			params_);
		eco_trainer_.train_joint();
		//assert(0);
		// 12. Update projection matrix P.
		projection_matrix_ = eco_trainer_.get_proj();
		
		// 13. Re-project the sample and update the sample space.
		xlf_proj = FeatureProjection(xlf, projection_matrix_);
		sample_update_.replace_sample(xlf_proj, 0); // put xlf_proj to the smaples_f_[0].

		// 14. Update distance matrix of sample space. Find the norm of the reprojected sample
		float new_sample_norm = FeatureComputeEnergy(xlf_proj);
		sample_update_.set_gram_matrix(0, 0, 2 * new_sample_norm);

		// 15. Update filter f.
		hf_full_ = full_fourier_coeff(eco_trainer_.get_hf());

	}

	bool ECO::update(const cv::Mat &frame, cv::Rect2f &roi, float score)
	{
		//**************************************************************************
		//*****                     Localization
		//**************************************************************************
		cv::Point sample_pos = (roi.width == -1) ? cv::Point(pos_) : cv::Point(roi.x, roi.y);
		//std::cout << "sample_pos = " << sample_pos << "pos_ = " << pos_ << ", roi = " << roi << std::endl;
		
		// 防止获得的pos点越界
		if (sample_pos.x > frame.cols - 2) sample_pos.x = frame.cols - 2;
		if (sample_pos.x < 2) sample_pos.x = 2;
		if (sample_pos.y > frame.rows - 2) sample_pos.y = frame.rows - 2;
		if (sample_pos.y < 2) sample_pos.y = 2;
		std::vector<float> samples_scales;
		//std::cout << "currentScaleFactor_ = " << currentScaleFactor_ << "scale_factors_ = [";
		for (size_t i = 0; i < scale_factors_.size(); ++i)
		{
			samples_scales.push_back(currentScaleFactor_ * scale_factors_[i]);
			//std::cout << scale_factors_[i] << ", ";
		}
		//std::cout << "]" << std::endl;

		// 1: Extract features at multiple resolutions
		ECO_FEATS xt = feature_extractor_.extractor(frame, sample_pos, samples_scales, params_, is_color_image_);
		if (xt[0].size() == 0)
		{
			//print too much will cause VOT_Trax fail.
			//debug("Feature window is zero.");
			return false;
		}
		//printf("xt size: %zu, %zu, %d x %d\n", xt.size(), xt[0].size(), xt[0][0].rows, xt[0][0].cols);

		// 2:  project sample
		ECO_FEATS xt_proj = FeatureProjectionMultScale(xt, projection_matrix_);
		//printf("xt_proj size: %zu, %zu, %d x %d\n", xt_proj.size(), xt_proj[0].size(), xt_proj[0][0].rows, xt_proj[0][0].cols);

		// 3: Do windowing of features
		xt_proj = do_windows(xt_proj, cos_window_);

		// 4: Compute the fourier series
		ECO_FEATS xtf_proj = do_dft(xt_proj);

		// 5: Interpolate features to the continuous domain
		xtf_proj = interpolate_dft(xtf_proj, interp1_fs_, interp2_fs_);
		//printf("xtf_proj size: %zu, %zu, %d x %d\n", xtf_proj.size(), xtf_proj[0].size(), xtf_proj[0][0].rows, xtf_proj[0][0].cols);
		
		// 6: Compute the scores in Fourier domain for different scales of target
		std::vector<cv::Mat> scores_fs_sum;
		for (size_t i = 0; i < scale_factors_.size(); i++) // for each scale
		{
			scores_fs_sum.push_back(cv::Mat::zeros(filter_size_[output_index_], CV_32FC2));
		}
			
		for (size_t i = 0; i < xtf_proj.size(); i++) // for each feature
		{
			int pad = (filter_size_[output_index_].height - xtf_proj[i][0].rows) / 2;
			cv::Rect temp_roi = cv::Rect(pad, pad, xtf_proj[i][0].cols, xtf_proj[i][0].rows); // get the roi
			
			for (size_t j = 0; j < xtf_proj[i].size(); j++)									  // for dimension x scale
			{
				size_t k1 = j / hf_full_[i].size(); // for each scale
				size_t k2 = j % hf_full_[i].size(); // for each dimension of scale
				//printf("%lu, %lu, %lu, %lu\n", i, j, k1, k2);
				scores_fs_sum[k1](temp_roi) += complexDotMultiplication(xtf_proj[i][j], hf_full_[i][k2]);
			}
		}

		//printf("scores_fs_sum: %zu, %d x %d\n", scores_fs_sum.size(), scores_fs_sum[0].rows, scores_fs_sum[0].cols);

		// 7: Calculate score by inverse DFT and
		// 8: Optimize the continuous score function with Newton's method.
		OptimizeScores scores(scores_fs_sum, params_.newton_iterations);
		scores.compute_scores();

		//std::cout << "scores = " << scores.get_max_score() << ", pos_ = " << pos_ << ", samples_scales = " << samples_scales[0] << std::endl;

		params_.max_score_threshhold = score;
		if (scores.get_max_score() >= params_.max_score_threshhold)
		{

			// Compute the translation vector in pixel-coordinates and round to the closest integer pixel.
			int scale_change_factor = scores.get_scale_ind();
			float resize_scores_width = (img_support_size_.width / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
			float resize_scores_height = (img_support_size_.height / output_size_) * currentScaleFactor_ * scale_factors_[scale_change_factor];
			float dx = scores.get_disp_col() * resize_scores_width;
			float dy = scores.get_disp_row() * resize_scores_height;
			//printf("scale_change_factor:%d, get_disp_col: %f, get_disp_row: %f, dx: %f, dy: %f\n", scale_change_factor, scores.get_disp_col(), scores.get_disp_row(), dx, dy);


			// 9: Update position
			pos_ = cv::Point2f(sample_pos) + cv::Point2f(dx, dy);


			// Do scale tracking with the scale filtertraffic
			if (nScales_ > 0 && params_.use_scale_filter)
			{
				scale_change_factor = scale_filter_.scale_filter_track(frame, pos_, base_target_size_, currentScaleFactor_, params_);
			}
			// Update the scale
			currentScaleFactor_ = currentScaleFactor_ * scale_factors_[scale_change_factor];

			// Adjust the scale to make sure we are not too large or too small
			if (currentScaleFactor_ < params_.min_scale_factor)
			{
				currentScaleFactor_ = params_.min_scale_factor;
			}
			else if (currentScaleFactor_ > params_.max_scale_factor)
			{
				currentScaleFactor_ = params_.max_scale_factor;
			}

			//**************************************************************************
			//*****                     Model update
			//**************************************************************************
			// 1: Get the sample calculated in localization
			ECO_FEATS xlf_proj;
			for (size_t i = 0; i < xtf_proj.size(); ++i)
			{
				std::vector<cv::Mat> tmp;
				int start_ind = scale_change_factor * projection_matrix_[i].cols;
				int end_ind = (scale_change_factor + 1) * projection_matrix_[i].cols;
				for (size_t j = start_ind; j < (size_t)end_ind; ++j)
				{
					tmp.push_back(xtf_proj[i][j].colRange(0,
						xtf_proj[i][j].rows / 2 + 1));
				}
				xlf_proj.push_back(tmp);
			}

			//printf("xlf_proj size: %zu, %zu, %d x %d\n", xlf_proj.size(), xlf_proj[0].size(), xlf_proj[0][0].rows, xlf_proj[0][0].cols);

			// 2: Shift the sample so that the target is centered,
			//  A shift in spatial domain means multiply by exp(i pi L k), according to shift property of Fourier transformation.
			cv::Point2f shift_samp =
				2.0f * CV_PI * cv::Point2f(pos_ - cv::Point2f(sample_pos)) *
				(1.0f / (currentScaleFactor_ * img_support_size_.width));
			xlf_proj = shift_sample(xlf_proj, shift_samp, kx_, ky_);
			//printf("shift_sample: %f %f\n", shift_samp.x, shift_samp.y);

			// 3: Update the samples space to include the new sample, the distance matrix,
			// kernel matrix and prior weight are also updated
			//debug("get_merged_sample_id: %d, get_new_sample_id: %d", sample_update_.get_merged_sample_id(), sample_update_.get_new_sample_id());
			sample_update_.update_sample_space_model(xlf_proj);

			// 4: Train the tracker every Nsth frame, Ns in ECO paper
			bool train_tracker = frames_since_last_train_ >= (size_t)params_.train_gap;

			// Set conjugate gradient options
			params_.CG_opts.CG_use_FR = params_.CG_use_FR;
			params_.CG_opts.tol = 1e-6;
			params_.CG_opts.CG_standard_alpha = params_.CG_standard_alpha;
			params_.CG_opts.debug = params_.debug;
			params_.CG_opts.maxit = params_.CG_iter;

			if (train_tracker)
			{
				//printf("train_tracker: %zu %zu\n", sample_energy_.size(), FeautreComputePower2(xlf_proj).size());
				sample_energy_ = sample_energy_ * (1 - params_.learning_rate) +
					FeautreComputePower2(xlf_proj) * params_.learning_rate;

				eco_trainer_.train_filter(sample_update_.get_samples(),
					sample_update_.get_prior_weights(),
					sample_energy_); // #6 x slower#

				frames_since_last_train_ = 0;
			}
			else
			{
				++frames_since_last_train_;
			}
			// 5: Update projection matrix P.
			// projection_matrix_ = eco_trainer_.get_proj();

			// 6: Update filter f.
			hf_full_ = full_fourier_coeff(eco_trainer_.get_hf());
		}

		//**************************************************************************
		//*****                    			Return
		//**************************************************************************
		roi.width = base_target_size_.width * currentScaleFactor_;
		roi.height = base_target_size_.height * currentScaleFactor_;
		roi.x = pos_.x - roi.width / 2;
		roi.y = pos_.y - roi.height / 2;

		return scores.get_max_score() >= params_.max_score_threshhold;
	}

	void ECO::init_parameters(const eco::EcoParameters &parameters)
	{
		// Features
		params_.useDeepFeature = parameters.useDeepFeature;
		params_.useHogFeature = parameters.useHogFeature;
		params_.useColorspaceFeature = parameters.useColorspaceFeature;
		params_.useCnFeature = parameters.useCnFeature;
		params_.useIcFeature = parameters.useIcFeature;

		params_.hog_features.fparams.cell_size = parameters.hog_features.fparams.cell_size;
		params_.cn_features.fparams.tablename = parameters.cn_features.fparams.tablename;
		//params_.ic_features.fparams.tablename = parameters.ic_features.fparams.tablename;

		// Extra parameters
		params_.max_score_threshhold = parameters.max_score_threshhold;

		// Global feature parameters1s
		params_.normalize_power = parameters.normalize_power;
		params_.normalize_size = parameters.normalize_size;
		params_.normalize_dim = parameters.normalize_dim;

		// img sample parameters
		params_.search_area_shape = parameters.search_area_shape;
		params_.search_area_scale = parameters.search_area_scale;
		params_.min_image_sample_size = parameters.min_image_sample_size;
		params_.max_image_sample_size = parameters.max_image_sample_size;

		// Detection parameters
		params_.refinement_iterations = parameters.refinement_iterations;
		params_.newton_iterations = parameters.newton_iterations;
		params_.clamp_position = parameters.clamp_position;

		// Learning parameters
		params_.output_sigma_factor = parameters.output_sigma_factor;
		params_.learning_rate = parameters.learning_rate;
		params_.sample_replace_strategy = parameters.sample_replace_strategy;
		params_.nSamples = parameters.nSamples;
		params_.lt_size = parameters.lt_size;
		params_.train_gap = parameters.train_gap;
		params_.skip_after_frame = parameters.skip_after_frame;
		params_.use_detection_sample = parameters.use_detection_sample;

		// Factorized convolution parameters
		params_.use_projection_matrix = parameters.use_projection_matrix;
		params_.update_projection_matrix = parameters.update_projection_matrix;
		params_.proj_init_method = parameters.proj_init_method;
		params_.projection_reg = parameters.projection_reg;

		// Generative sample space model parameters
		params_.use_sample_merge = parameters.use_sample_merge;
		params_.sample_merge_type = parameters.sample_merge_type;
		params_.distance_matrix_update_type = parameters.distance_matrix_update_type;

		// Conjugate Gradient parameters
		params_.CG_iter = parameters.CG_iter;
		params_.init_CG_iter = parameters.init_CG_iter;
		params_.init_GN_iter = parameters.init_GN_iter;
		params_.CG_use_FR = parameters.CG_use_FR;
		params_.CG_standard_alpha = parameters.CG_standard_alpha;
		params_.CG_forgetting_rate = parameters.CG_forgetting_rate;
		params_.precond_data_param = parameters.precond_data_param;
		params_.precond_reg_param = parameters.precond_reg_param;
		params_.precond_proj_param = parameters.precond_proj_param;

		// Regularization window parameters
		params_.use_reg_window = parameters.use_reg_window;
		params_.reg_window_min = parameters.reg_window_min;
		params_.reg_window_edge = parameters.reg_window_edge;
		params_.reg_window_power = parameters.reg_window_power;
		params_.reg_sparsity_threshold = parameters.reg_sparsity_threshold;

		// Interpolation parameters
		params_.interpolation_method = parameters.interpolation_method;
		params_.interpolation_bicubic_a = parameters.interpolation_bicubic_a;
		params_.interpolation_centering = parameters.interpolation_centering;
		params_.interpolation_windowing = parameters.interpolation_windowing;

		// Scale parameters for the translation model
		params_.number_of_scales = parameters.number_of_scales;
		params_.scale_step = parameters.scale_step;

		// Scale filter parameters
		params_.use_scale_filter = parameters.use_scale_filter;
		params_.scale_sigma_factor = parameters.scale_sigma_factor;
		params_.scale_learning_rate = parameters.scale_learning_rate;
		params_.number_of_scales_filter = parameters.number_of_scales_filter;
		params_.number_of_interp_scales = parameters.number_of_interp_scales;
		params_.scale_model_factor = parameters.scale_model_factor;
		params_.scale_step_filter = parameters.scale_step_filter;
		params_.scale_model_max_area = parameters.scale_model_max_area;
		params_.scale_feature = parameters.scale_feature;
		params_.s_num_compressed_dim = parameters.s_num_compressed_dim;
		params_.lambda = parameters.lambda;
		params_.do_poly_interp = parameters.do_poly_interp;

		params_.debug = parameters.debug;

		// GPU
		//params_.use_gpu = parameters.use_gpu;
		//params_.gpu_id = parameters.gpu_id;
	}

	void ECO::init_features()
	{
		params_.useDeepFeature = false;
		if (!params_.useDeepFeature)
		{
			img_support_size_ = img_sample_size_;
		}

		if (params_.useHogFeature) // just HOG feature;
		{
			params_.hog_features.img_input_sz = img_support_size_;
			params_.hog_features.img_sample_sz = img_support_size_;
			params_.hog_features.data_sz_block0 = cv::Size(
				params_.hog_features.img_sample_sz.width /
				params_.hog_features.fparams.cell_size,
				params_.hog_features.img_sample_sz.height /
				params_.hog_features.fparams.cell_size);
		}
		if (params_.useColorspaceFeature)
		{
		}
		if (params_.useCnFeature && is_color_image_)
		{
			params_.cn_features.img_input_sz = img_support_size_;
			params_.cn_features.img_sample_sz = img_support_size_;
			params_.cn_features.data_sz_block0 = cv::Size(
				params_.cn_features.img_sample_sz.width /
				params_.cn_features.fparams.cell_size,
				params_.cn_features.img_sample_sz.height /
				params_.cn_features.fparams.cell_size);

			// 数据深拷贝
			size_t rows = sizeof(params_.cn_features.fparams.table) / sizeof(params_.cn_features.fparams.table[0]);
			size_t cols = sizeof(params_.cn_features.fparams.table[0]) / sizeof(float);
			std::copy(&cn_norm_table[0][0], &cn_norm_table[0][0] + rows * cols, &params_.cn_features.fparams.table[0][0]);

			// 现在的电脑在执行exe读取txt时，读取的数据是有问题的，因此不能再执行读txt的操作
			/*
			std::string s;
			std::string path = params_.cn_features.fparams.tablename;
			std::ifstream *read = new std::ifstream(path);
			size_t rows = sizeof(params_.cn_features.fparams.table) / sizeof(params_.cn_features.fparams.table[0]);
			size_t cols = sizeof(params_.cn_features.fparams.table[0]) / sizeof(float);
			//debug("rows:%lu,cols:%lu", rows, cols);
			std::fstream f("cn_norm.txt", std::ios::out);
			f << "{ \n";
			for (size_t i = 0; i < rows; i++)
			{
				f << "{";
				for (size_t j = 0; j < cols - 1; j++)
				{
					std::getline(*read, s, '\t');
					params_.cn_features.fparams.table[i][j] = atof(s.c_str());
					f << s << ", ";
				}
				std::getline(*read, s);
				params_.cn_features.fparams.table[i][cols - 1] = atof(s.c_str());
				f << s << "}, \n";
			}
			f << "}";
			f.close();*/
		}
	
		// features setting-----------------------------------------------------
		if (params_.useHogFeature)
		{
			feature_size_.push_back(params_.hog_features.data_sz_block0);
			feature_dim_.push_back(params_.hog_features.fparams.nDim);
			compressed_dim_.push_back(params_.hog_features.fparams.compressed_dim);
		}
		if (params_.useColorspaceFeature)
		{
		}
		if (params_.useCnFeature && is_color_image_)
		{
			feature_size_.push_back(params_.cn_features.data_sz_block0);
			feature_dim_.push_back(params_.cn_features.fparams.nDim);
			compressed_dim_.push_back(params_.cn_features.fparams.compressed_dim);
		}
	}

	void ECO::yf_gaussian() // real part of (9) in paper C-COT 特征的期望输出，C-COT论文公式9的实数部分
	{
		// sig_y is sigma in (9)
		double sig_y = sqrt(int(base_target_size_.width) *
			int(base_target_size_.height)) *
			(params_.output_sigma_factor) *
			(float(output_size_) / img_support_size_.width);

		double tmp1 = CV_PI * sig_y / output_size_;
		double tmp2 = std::sqrt(2 * CV_PI) * sig_y / output_size_;
		for (unsigned int i = 0; i < ky_.size(); i++) // for each filter
		{
			// 2 dimension version of (9)
			/*
			cv::Mat tempy(ky_[i].size(), CV_32FC1);
			tempy = tmp1 * ky_[i];
			cv::exp(-2 * tempy.mul(tempy), tempy);
			tempy = tmp2 * tempy;

			cv::Mat tempx(kx_[i].size(), CV_32FC1);
			tempx = tmp1 * kx_[i];
			cv::exp(-2 * tempx.mul(tempx), tempx);
			tempx = tmp2 * tempx;

			yf_.push_back(cv::Mat(tempy * tempx)); // matrix multiplication
	*/

			cv::Mat temp(ky_[i].rows, kx_[i].cols, CV_32FC1);
			for (int r = 0; r < temp.rows; r++)
			{
				float tempy = tmp1 * ky_[i].at<float>(r, 0);
				tempy = tmp2 * std::exp(-2.0f * tempy * tempy);
				for (int c = 0; c < temp.cols; c++)
				{
					float tempx = tmp1 * kx_[i].at<float>(0, c);
					tempx = tmp2 * std::exp(-2.0f * tempx * tempx);
					temp.at<float>(r, c) = tempy * tempx;
				}
			}

			//        plt::plot_surface_(temp);

			//        plt::imshow(temp.data, temp.rows, temp.cols, 1);
			//        plt::show();

			yf_.push_back(temp);

		}
	}

	void ECO::cos_window()
	{
		for (size_t i = 0; i < feature_size_.size(); i++)
		{
			/*
			cv::Mat hann1t = cv::Mat(cv::Size(feature_size_[i].width + 2, 1), CV_32F, cv::Scalar(0));
			cv::Mat hann2t = cv::Mat(cv::Size(1, feature_size_[i].height + 2), CV_32F, cv::Scalar(0));
			for (int i = 0; i < hann1t.cols; i++)
				hann1t.at<float>(0, i) = 0.5 * (1 - std::cos(2 * M_PI * i / (hann1t.cols - 1)));
			for (int i = 0; i < hann2t.rows; i++)
				hann2t.at<float>(i, 0) = 0.5 * (1 - std::cos(2 * M_PI * i / (hann2t.rows - 1)));
			cv::Mat hann2d = hann2t * hann1t;
			cos_window_.push_back(hann2d(cv::Range(1, hann2d.rows - 1),
										 cv::Range(1, hann2d.cols - 1)));
			*/
			cv::Mat temp(feature_size_[i].height, feature_size_[i].width, CV_32FC1);
			//cv::Mat temp(feature_size_[i].height + 2, feature_size_[i].width + 2, CV_32FC1);
			for (int r = 0; r < temp.rows; r++)
			{
				float tempy = 0.5f * (1 - std::cos(2 * CV_PI * (float)(r + 1) / (feature_size_[i].width + 1)));
				for (int c = 0; c < temp.cols; c++)
				{
					temp.at<float>(r, c) = tempy * 0.5f * (1 - std::cos(2 * (float)(c + 1) * CV_PI / (feature_size_[i].height + 1)));
				}
			}
			//        plt::plot_surface_(temp);
			cos_window_.push_back(temp);
		}
	}

	ECO_FEATS ECO::interpolate_dft(const ECO_FEATS &xlf, std::vector<cv::Mat> &interp1_fs, std::vector<cv::Mat> &interp2_fs)
	{
		ECO_FEATS result;
		for (size_t i = 0; i < xlf.size(); i++)
		{
			cv::Mat interp1_fs_mat =
				subwindow(interp1_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp1_fs[i].rows, interp1_fs[i].rows)), IPL_BORDER_REPLICATE);
			cv::Mat interp2_fs_mat =
				subwindow(interp2_fs[i], cv::Rect(cv::Point(0, 0), cv::Size(interp2_fs[i].cols, interp2_fs[i].cols)), IPL_BORDER_REPLICATE);
			std::vector<cv::Mat> temp;

			for (size_t j = 0; j < xlf[i].size(); j++)
			{
				temp.push_back(complexDotMultiplication(
					complexDotMultiplication(interp1_fs_mat, xlf[i][j]), interp2_fs_mat));
			}
			result.push_back(temp);
		}
		return result;
	}

	// Take half of the fourier coefficient.
	ECO_FEATS ECO::compact_fourier_coeff(const ECO_FEATS &xf)
	{
		ECO_FEATS result;
		for (size_t i = 0; i < xf.size(); i++) // for each feature
		{
			std::vector<cv::Mat> temp;
			for (size_t j = 0; j < xf[i].size(); j++) // for each dimension
				temp.push_back(xf[i][j].colRange(0, (xf[i][j].cols + 1) / 2));
			result.push_back(temp);
		}
		return result;
	}

	// Get the full fourier coefficient of xf, using the property X(N-k)=conv(X(k))
	ECO_FEATS ECO::full_fourier_coeff(const ECO_FEATS &xf)
	{
		ECO_FEATS res;
		for (size_t i = 0; i < xf.size(); i++) // for each feature
		{
			std::vector<cv::Mat> tmp;
			for (size_t j = 0; j < xf[i].size(); j++) // for each dimension
			{
				cv::Mat temp = xf[i][j].colRange(0, xf[i][j].cols - 1).clone();
				//std::cout << "temp1 = " << temp << std::endl;
				rot90(temp, 3);
				cv::hconcat(xf[i][j], mat_conj(temp), temp);
				//std::cout << "temp2 = " << temp << std::endl;
				tmp.push_back(temp);
			}
			res.push_back(tmp);
		}

		return res;
	}

	std::vector<cv::Mat> ECO::project_mat_energy(std::vector<cv::Mat> proj,
		std::vector<cv::Mat> yf)
	{
		std::vector<cv::Mat> result;
		for (size_t i = 0; i < yf.size(); i++)
		{
			cv::Mat temp(proj[i].size(), CV_32FC1);
			float sum_dim = std::accumulate(feature_dim_.begin(),
				feature_dim_.end(),
				0.0f);
			cv::Mat x = yf[i].mul(yf[i]);
			temp = 2 * mat_sum_f(x) / sum_dim *
				cv::Mat::ones(proj[i].size(), CV_32FC1);
			result.push_back(temp);
		}
		return result;
	}

	// Shift a sample in the Fourier domain. The shift should be normalized to the range [-pi, pi]
	ECO_FEATS ECO::shift_sample(ECO_FEATS &xf,
		cv::Point2f shift,
		std::vector<cv::Mat> kx,
		std::vector<cv::Mat> ky)
	{
		ECO_FEATS res;

		for (size_t i = 0; i < xf.size(); ++i) // for each feature
		{
			cv::Mat shift_exp_y(ky[i].size(), CV_32FC2);
			cv::Mat shift_exp_x(kx[i].size(), CV_32FC2);
			for (size_t j = 0; j < (size_t)ky[i].rows; j++)
			{
				shift_exp_y.at<COMPLEX>(j, 0) = COMPLEX(cos(shift.y * ky[i].at<float>(j, 0)), sin(shift.y * ky[i].at<float>(j, 0)));
			}
			for (size_t j = 0; j < (size_t)kx[i].cols; j++)
			{
				shift_exp_x.at<COMPLEX>(0, j) = COMPLEX(cos(shift.x * kx[i].at<float>(0, j)), sin(shift.x * kx[i].at<float>(0, j)));
			}
			/*
			debug("shift_exp_y:");
			showmat2chall(shift_exp_y, 2);
			debug("shift_exp_x:");
			showmat2chall(shift_exp_x, 2);
			assert(0);
			*/
			cv::Mat shift_exp_y_mat =
				subwindow(shift_exp_y, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);
			cv::Mat shift_exp_x_mat =
				subwindow(shift_exp_x, cv::Rect(cv::Point(0, 0), xf[i][0].size()), IPL_BORDER_REPLICATE);

			std::vector<cv::Mat> tmp;
			for (size_t j = 0; j < xf[i].size(); j++) // for each dimension of the feature, do complex element-wise multiplication
			{
				tmp.push_back(complexDotMultiplication(
					complexDotMultiplication(shift_exp_y_mat, xf[i][j]), shift_exp_x_mat));
			}
			res.push_back(tmp);
		}
		return res;
	}
} // namespace eco
