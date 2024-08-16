#include "feature_extractor.h"
#include "recttools.h"
#include "fhog.h"
#include <opencv2/core/core_c.h>

namespace eco
{
	ECO_FEATS FeatureExtractor::extractor(const cv::Mat image,
		const cv::Point2f pos,
		const std::vector<float> scales,
		const EcoParameters &params,
		const bool &is_color_image)
	{
		params_ = params;
		int num_features = 0, num_scales = scales.size();
		std::vector<cv::Size2f> img_sample_sz;
		std::vector<cv::Size2f> img_input_sz;

		if (params.useHogFeature)
		{
			hog_feat_ind_ = num_features;
			num_features++;
			hog_features_ = params.hog_features;
			img_sample_sz.push_back(hog_features_.img_sample_sz);
			img_input_sz.push_back(hog_features_.img_input_sz);
		}

		if (params.useCnFeature && is_color_image)
		{
			cn_feat_ind_ = num_features;
			num_features++;
			cn_features_ = params.cn_features;
			img_sample_sz.push_back(cn_features_.img_sample_sz);
			img_input_sz.push_back(cn_features_.img_input_sz);
		}
		
		// Extract images for different feautures --------------------------
		std::vector<std::vector<cv::Mat>> img_samples;
		for (int i = 0; i < num_features; ++i) // for each feature
		{
			std::vector<cv::Mat> img_samples_temp(num_scales);
			for (unsigned int j = 0; j < scales.size(); ++j) // for each scale
			{
				img_samples_temp[j] = sample_patch(image, pos, img_sample_sz[i] * scales[j], img_input_sz[i]);
			}
			img_samples.push_back(img_samples_temp);
		}

		// Extract features ------------------------------------------------
		ECO_FEATS sum_features;

		if (params.useHogFeature)
		{
			hog_feat_maps_ = get_hog_features(img_samples[hog_feat_ind_]);
			hog_feat_maps_ = hog_feature_normalization(hog_feat_maps_);
			sum_features.push_back(hog_feat_maps_);
		}

		if (params.useCnFeature && is_color_image)
		{
			cn_feat_maps_ = get_cn_features(img_samples[cn_feat_ind_]);
			cn_feat_maps_ = cn_feature_normalization(cn_feat_maps_);
			sum_features.push_back(cn_feat_maps_);
		}

		return sum_features;
	}

	cv::Mat FeatureExtractor::sample_patch(const cv::Mat im,
		const cv::Point2f posf,
		cv::Size2f sample_sz,
		cv::Size2f input_sz)
	{
		// Pos should be integer when input, but floor in just in case.
		cv::Point2i pos(posf);
		//debug("%d, %d", pos.y, pos.x);

		// Downsample factor
		float resize_factor = std::min(sample_sz.width / input_sz.width,
			sample_sz.height / input_sz.height);
		int df = std::max((float)floor(resize_factor - 0.1), float(1));
		//debug("resize_factor: %f, df: %d,sample_sz: %f x %f,input_sz: % f x % f",
		//	  resize_factor, df,
		//	  sample_sz.width, sample_sz.height,
		//	  input_sz.width, input_sz.height);

		cv::Mat new_im;
		im.copyTo(new_im);
		//debug("new_im:%d x %d", new_im.rows, new_im.cols);

		if (df > 1)
		{
			// compute offset and new center position
			cv::Point os((pos.x - 1) % df, ((pos.y - 1) % df));
			pos.x = (pos.x - os.x - 1) / df + 1;
			pos.y = (pos.y - os.y - 1) / df + 1;
			// new sample size
			sample_sz.width = sample_sz.width / df;
			sample_sz.height = sample_sz.height / df;
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

		// make sure the size is not too small and round it
		sample_sz.width = std::max(round(sample_sz.width), 2.0f);
		sample_sz.height = std::max(round(sample_sz.height), 2.0f);

		cv::Point pos2(pos.x - floor((sample_sz.width + 1) / 2),
			pos.y - floor((sample_sz.height + 1) / 2));
		//debug("new_im:%d x %d, pos2:%d %d, sample_sz:%f x %f", new_im.rows, new_im.cols, pos2.y, pos2.x, sample_sz.height, sample_sz.width);

		cv::Mat im_patch = subwindow(new_im, cv::Rect(pos2, sample_sz), IPL_BORDER_REPLICATE);

		cv::Mat resized_patch;
		if (im_patch.cols == 0 || im_patch.rows == 0)
		{
			return resized_patch;
		}
		cv::resize(im_patch, resized_patch, input_sz);
		/* Debug
		printMat(resized_patch); // 8UC3 150 x 150
		showmat3ch(resized_patch, 0);
		// resized_patch(121,21,1) in matlab RGB, cv: BGR
		debug("%d", resized_patch.at<cv::Vec3b>(120,20)[2]);
		assert(0); */
		return resized_patch;
	}

	std::vector<cv::Mat> FeatureExtractor::get_hog_features(const std::vector<cv::Mat> ims)
	{
		if (ims.empty())
		{
			return std::vector<cv::Mat>();
		}
		std::vector<cv::Mat> hog_feats;
		for (unsigned int i = 0; i < ims.size(); i++)
		{
			cv::Mat ims_f;
			ims[i].convertTo(ims_f, CV_32FC3);

			cv::Size _tmpl_sz;
			_tmpl_sz.width = ims_f.cols;
			_tmpl_sz.height = ims_f.rows;

			int _cell_size = hog_features_.fparams.cell_size;
			// Round to cell size and also make it even
			if (int(_tmpl_sz.width / (_cell_size)) % 2 == 0)
			{
				_tmpl_sz.width = ((int)(_tmpl_sz.width / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 2;
				_tmpl_sz.height = ((int)(_tmpl_sz.height / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 2;
			}
			else
			{
				_tmpl_sz.width = ((int)(_tmpl_sz.width / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 3;
				_tmpl_sz.height = ((int)(_tmpl_sz.height / (2 * _cell_size)) * 2 * _cell_size) + _cell_size * 3;
			}

			// Add extra cell filled with zeros around the image
			cv::Mat featurePaddingMat(_tmpl_sz.height + _cell_size * 2,
				_tmpl_sz.width + _cell_size * 2,
				CV_32FC3, cvScalar(0, 0, 0));

			if (ims_f.cols != _tmpl_sz.width || ims_f.rows != _tmpl_sz.height)
			{
				resize(ims_f, ims_f, _tmpl_sz);
			}
			ims_f.copyTo(featurePaddingMat);

			IplImage zz = cvIplImage(featurePaddingMat);
			CvLSVMFeatureMapCaskade *map_temp;
			getFeatureMaps(&zz, _cell_size, &map_temp); // dimension: 27
			normalizeAndTruncate(map_temp, 0.2f);		// dimension: 108
			PCAFeatureMaps(map_temp);					// dimension: 31

			// Procedure do deal with cv::Mat multichannel bug(can not merge)
			cv::Mat featuresMap = cv::Mat(cv::Size(map_temp->sizeX, map_temp->sizeY), CV_32FC(map_temp->numFeatures), map_temp->map);

			// clone because map_temp will be free.
			featuresMap = featuresMap.clone();

			freeFeatureMapObject(&map_temp);

			hog_feats.push_back(featuresMap);
		}

		return hog_feats;
	}

	std::vector<cv::Mat> FeatureExtractor::hog_feature_normalization(std::vector<cv::Mat> &hog_feat_maps)
	{
		if (hog_feat_maps.empty())
		{
			return std::vector<cv::Mat>();
		}
		std::vector<cv::Mat> hog_maps_vec;
		for (size_t i = 0; i < hog_feat_maps.size(); i++)
		{
			if (hog_feat_maps[i].cols == 0 || hog_feat_maps[i].rows == 0)
			{
				std::vector<cv::Mat> emptyMat;
				hog_maps_vec.insert(hog_maps_vec.end(), emptyMat.begin(), emptyMat.end());
			}
			else
			{
				cv::Mat temp = hog_feat_maps[i].mul(hog_feat_maps[i]);
				// float sum_scales = cv::sum(temp)[0]; // sum can not work when dimension exceeding 3
				std::vector<cv::Mat> temp_vec, result_vec;
				float sum = 0;
				cv::split(temp, temp_vec);
				for (int j = 0; j < temp.channels(); j++)
				{
					sum += cv::sum(temp_vec[j])[0];
				}
				float para = hog_features_.data_sz_block0.area() * hog_features_.fparams.nDim;
				hog_feat_maps[i] *= sqrt(para / sum);
				//debug("para:%f, sum:%f, sqrt:%f", para, sum, sqrt(para / sum));
				cv::split(hog_feat_maps[i], result_vec);
				hog_maps_vec.insert(hog_maps_vec.end(), result_vec.begin(), result_vec.end());
			}
		}
		return hog_maps_vec;
	}

	//=========================================================================
	std::vector<cv::Mat> FeatureExtractor::get_cn_features(const std::vector<cv::Mat> ims)
	{
		if (ims.empty())
		{
			return std::vector<cv::Mat>();
		}

		std::vector<cv::Mat> cn_feats;
		float den = 8.0f, fac = 32.0f;
		for (size_t i = 0; i < ims.size(); i++)
		{
			// table_lookup()-----------------------------------
			cv::Mat ims_f, index_im;
			if (ims[i].channels() == 3)
			{
				ims[i].convertTo(ims_f, CV_32FC3);

				ims_f /= den;
				std::vector<cv::Mat> ims_vector;
				cv::split(ims_f, ims_vector);
				for (int i = 0; i < ims_f.rows; i++)
					for (int j = 0; j < ims_f.cols; j++)
						for (int k = 0; k < 3; k++)
						{
							ims_vector[k].at<float>(i, j) = std::floor(ims_vector[k].at<float>(i, j));
						}
				// matlab: RGB, opencv:BGR
				index_im = ims_vector[2] + fac * ims_vector[1] + fac * fac * ims_vector[0];
			}
			else
			{
				ims[i].convertTo(ims_f, CV_32FC1);
				ims_f /= den;
				for (int i = 0; i < ims_f.rows; i++)
					for (int j = 0; j < ims_f.cols; j++)
					{
						ims_f.at<float>(i, j) = std::floor(ims_f.at<float>(i, j));
					}
				index_im = ims_f;
			}

			const int Layers = sizeof(params_.cn_features.fparams.table[0]) / sizeof(float);
			cv::Mat tableMap = cv::Mat(ims_f.size(), CV_32FC(Layers));
			for (int i = 0; i < ims_f.rows; i++)
				for (int j = 0; j < ims_f.cols; j++)
					for (int k = 0; k < Layers; k++)
					{
						tableMap.at<cv::Vec<float, Layers>>(i, j)[k] = params_.cn_features.fparams.table[(size_t)index_im.at<float>(i, j)][k];
					}

			// average_feature_region()-----------------------
			// integralVecImage()
			cv::Mat iImage_tmp = cv::Mat::zeros(ims_f.rows + 1, ims_f.cols + 1, CV_32FC(Layers));
			for (int i = 1; i < iImage_tmp.rows; i++)
				for (int j = 1; j < iImage_tmp.cols; j++)
					for (int k = 0; k < Layers; k++)
					{
						iImage_tmp.at<cv::Vec<float, Layers>>(i, j)[k] = iImage_tmp.at<cv::Vec<float, Layers>>(i - 1, j)[k] + tableMap.at<cv::Vec<float, Layers>>(i - 1, j - 1)[k];
					}
			cv::Mat iImage = cv::Mat::zeros(ims_f.rows + 1, ims_f.cols + 1, CV_32FC(Layers));
			for (int i = 1; i < iImage.rows; i++)
				for (int j = 1; j < iImage.cols; j++)
					for (int k = 0; k < Layers; k++)
					{
						iImage.at<cv::Vec<float, Layers>>(i, j)[k] = iImage.at<cv::Vec<float, Layers>>(i, j - 1)[k] + iImage_tmp.at<cv::Vec<float, Layers>>(i, j)[k];
					}

			int cell = params_.cn_features.fparams.cell_size;
			float region_area = cell * cell;
			float maxval = 1.0f;
			cv::Mat featuresMap = cv::Mat(params_.cn_features.data_sz_block0, CV_32FC(Layers));
			for (int i = 0; i < featuresMap.rows; i++)
				for (int j = 0; j < featuresMap.cols; j++)
					for (int k = 0; k < Layers; k++)
					{
						int ii = i * cell;
						int jj = j * cell;
						featuresMap.at<cv::Vec<float, Layers>>(i, j)[k] =
							(iImage.at<cv::Vec<float, Layers>>(ii, jj)[k] - iImage.at<cv::Vec<float, Layers>>(ii + cell, jj)[k] - iImage.at<cv::Vec<float, Layers>>(ii, jj + cell)[k] + iImage.at<cv::Vec<float, Layers>>(ii + cell, jj + cell)[k]) / (region_area * maxval);
					}


			cn_feats.push_back(featuresMap);
		}
		return cn_feats;
	}

	std::vector<cv::Mat> FeatureExtractor::cn_feature_normalization(std::vector<cv::Mat> &cn_feat_maps)
	{
		if (cn_feat_maps.empty())
		{
			return std::vector<cv::Mat>();
		}
		std::vector<cv::Mat> cn_maps_vec;
		for (size_t i = 0; i < cn_feat_maps.size(); i++)
		{
			if (cn_feat_maps[i].cols == 0 || cn_feat_maps[i].rows == 0)
			{
				std::vector<cv::Mat> emptyMat;
				cn_feat_maps.insert(cn_feat_maps.end(), emptyMat.begin(), emptyMat.end());
			}
			else
			{
				cv::Mat temp = cn_feat_maps[i].mul(cn_feat_maps[i]);
				// float sum_scales = cv::sum(temp)[0]; // sum can not work when dimension exceeding 3
				std::vector<cv::Mat> temp_vec, result_vec;
				float sum = 0;
				cv::split(temp, temp_vec);
				for (int j = 0; j < temp.channels(); j++)
				{
					sum += cv::sum(temp_vec[j])[0];
				}
				float para = cn_features_.data_sz_block0.area() * cn_features_.fparams.nDim;
				cn_feat_maps[i] *= sqrt(para / sum);

				cv::split(cn_feat_maps[i], result_vec);
				cn_maps_vec.insert(cn_maps_vec.end(), result_vec.begin(), result_vec.end());
			}
		}
		//printMat(cn_maps_vec[9]);
		//showmat1channels(cn_maps_vec[9], 2);
		return cn_maps_vec;
	}

	//=========================================================================
} // namespace eco
