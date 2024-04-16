#include "scale_filter.h"
#include "ffttools.h"
#include "recttools.h"
#include "feature_extractor.h"

namespace eco
{
	void ScaleFilter::init(int &nScales, float &scale_step, const EcoParameters &params)
	{
		nScales = params.number_of_scales_filter;
		scale_step = params.scale_step_filter;
		float scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor;
		std::vector<float> scale_exp, scale_exp_shift;
		int scalemin = floor((1.0 - (float)nScales) / 2.0);
		int scalemax = floor(((float)nScales - 1.0) / 2.0);
		for (int i = scalemin; i <= scalemax; i++)
		{
			scale_exp.push_back(i * params.number_of_interp_scales / (float)nScales);
		}
		for (int i = 0; i < nScales; i++)
		{
			scale_exp_shift.push_back(scale_exp[(i + nScales / 2) % nScales]);
		}
	
		std::vector<float> interp_scale_exp, interp_scale_exp_shift;
		scalemin = floor((1.0 - (float)params.number_of_interp_scales) / 2.0);
		scalemax = floor(((float)params.number_of_interp_scales - 1.0) / 2.0);
		for (int i = scalemin; i <= scalemax; i++)
		{
			interp_scale_exp.push_back(i);
		}
		for (int i = 0; i < params.number_of_interp_scales; i++)
		{
			interp_scale_exp_shift.push_back(interp_scale_exp[(i + params.number_of_interp_scales / 2) % params.number_of_interp_scales]);
		}
		
		for (int i = 0; i < nScales; i++)
		{
			scaleSizeFactors_.push_back(std::pow(scale_step, scale_exp[i]));
		}
	
		for (int i = 0; i < params.number_of_interp_scales; i++)
		{
			interpScaleFactors_.push_back(std::pow(scale_step, interp_scale_exp_shift[i]));
		}
		

		cv::Mat ys_mat = cv::Mat(cv::Size(nScales, 1), CV_32FC1);
		for (int i = 0; i < nScales; i++)
		{
			ys_mat.at<float>(0, i) = std::exp(-0.5f * scale_exp_shift[i] * scale_exp_shift[i] / scale_sigma / scale_sigma);
		}
		
		yf_ = real(dft(ys_mat, false));

		for (int i = 0; i < nScales; i++)
		{
			window_.push_back(0.5f * (1.0f - std::cos(2 * CV_PI * i / (nScales - 1.0f))));
		}
	}

	float ScaleFilter::scale_filter_track(const cv::Mat &im, const cv::Point2f &pos, const cv::Size2f &base_target_sz, const float &currentScaleFactor, const EcoParameters &params)
	{
		std::vector<float> scales;
		for (unsigned int i = 0; i < scaleSizeFactors_.size(); i++)
		{
			scales.push_back(scaleSizeFactors_[i] * currentScaleFactor);
		}
		cv::Mat xs = extract_scale_sample(im, pos, base_target_sz, scales, params.scale_model_sz);

		//debug("Not finished!-------------------");
		assert(0);

		float scale_change_factor = 0.0;
		return scale_change_factor;
	}

	cv::Mat ScaleFilter::extract_scale_sample(const cv::Mat &im, const cv::Point2f &posf, const cv::Size2f &base_target_sz, std::vector<float> &scaleFactors, const cv::Size &scale_model_sz)
	{
		cv::Point2i pos(posf);
		int nScales = scaleFactors.size();
		int df = std::floor(*std::min_element(std::begin(scaleFactors), std::end(scaleFactors)));

		cv::Mat new_im;
		im.copyTo(new_im);
		if (df > 1)
		{
			// compute offset and new center position
			cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
			pos.x = (pos.x - os.x - 1) / df + 1;
			pos.y = (pos.y - os.y - 1) / df + 1;

			for (unsigned int i = 0; i < scaleFactors.size(); i++)
			{
				scaleFactors[i] /= df;
			}
			// down sample image
			int r = (im.rows - os.y) / df + 1;
			int c = (im.cols - os.x) / df;
			cv::Mat new_im2(r, c, im.type());
			new_im = new_im2;
			for (size_t i = 0 + os.y, m = 0;
				i < (size_t)im.rows && m < (size_t)new_im.rows;
				i += df, ++m)
			{
				for (size_t j = 0 + os.x, n = 0;
					j < (size_t)im.cols && n < (size_t)new_im.cols;
					j += df, ++n)
				{

					if (im.channels() == 1)
					{
						new_im.at<uchar>(m, n) = im.at<uchar>(i, j);
					}
					else
					{
						new_im.at<cv::Vec3b>(m, n) = im.at<cv::Vec3b>(i, j);
					}
				}
			}
		}

		for (int s = 0; s < nScales; s++)
		{
			cv::Size patch_sz;
			patch_sz.width = std::max(std::floor(base_target_sz.width * scaleFactors[s]), 2.0f);
			patch_sz.height = std::max(std::floor(base_target_sz.height * scaleFactors[s]), 2.0f);
			//debug("patch_sz:%d %d", patch_sz.height, patch_sz.width);

			cv::Point pos2(pos.x - floor((patch_sz.width + 1) / 2),
				pos.y - floor((patch_sz.height + 1) / 2));

			cv::Mat im_patch = subwindow(new_im, cv::Rect(pos2, patch_sz), IPL_BORDER_REPLICATE);

			cv::Mat im_patch_resized;
			if (im_patch.cols == 0 || im_patch.rows == 0)
			{
				return im_patch_resized;
			}
			cv::resize(im_patch, im_patch_resized, scale_model_sz);

			std::vector<cv::Mat> im_vector, temp_hog;
			im_vector.push_back(im_patch);
			FeatureExtractor feature_extractor;

			temp_hog = feature_extractor.get_hog_features(im_vector);

			temp_hog = feature_extractor.hog_feature_normalization(temp_hog);
			assert(0);
		}

		cv::Mat scale_sample;
		return scale_sample;
	}

} // namespace eco
