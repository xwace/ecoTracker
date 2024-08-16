#include "ffttools.h"
namespace eco
{
	cv::Mat dft(const cv::Mat img_org, const bool backwards)
	{
		if (img_org.empty())
			return cv::Mat();
		cv::Mat img;
		int type = img_org.type() & CV_MAT_DEPTH_MASK;
		//debug("%d", type);//5:float;6:double
		assert(((type == 5) || (type == 6)) && "error: input mat type error!");

		img_org.copyTo(img);
		if (img.channels() == 1)
		{
			if (type == 5)
			{
				cv::Mat planes[] = { cv::Mat_<float>(img),
									cv::Mat_<float>::zeros(img.size()) };
				cv::merge(planes, 2, img);
			}
			else if (type == 6)
			{
				cv::Mat planes[] = { cv::Mat_<double>(img),
									cv::Mat_<double>::zeros(img.size()) };
				cv::merge(planes, 2, img);
			}
		}
		cv::dft(img, img, backwards ? (cv::DFT_INVERSE + cv::DFT_SCALE) : 0);

		return img;
	} // namespace eco

	cv::Mat fftshift(const cv::Mat img_org,
		const bool rowshift,
		const bool colshift,
		const bool reverse)
	{
		if (img_org.empty())
			return cv::Mat();
		int type = img_org.type() & CV_MAT_DEPTH_MASK;
		//debug("%d", type);//5:float;6:double

		assert(((type == 5) || (type == 6)) && "error: input mat type error!");

		cv::Mat temp(img_org.size(), img_org.type());

		int w = img_org.cols, h = img_org.rows;
		int rshift = reverse ? h - h / 2 : h / 2,
			cshift = reverse ? w - w / 2 : w / 2;

		for (int i = 0; i < img_org.rows; i++)
		{
			int ii = rowshift ? (i + rshift) % h : i;
			for (int j = 0; j < img_org.cols; j++)
			{
				int jj = colshift ? (j + cshift) % w : j;
				if (type == 5)
				{
					if (img_org.channels() == 2)
						temp.at<cv::Vec<float, 2>>(ii, jj) =
						img_org.at<cv::Vec<float, 2>>(i, j);
					else if (img_org.channels() == 1)
						temp.at<float>(ii, jj) = img_org.at<float>(i, j);
					else
						assert(0 && "error of image channels.");
				}
				else if (type == 6)
				{
					if (img_org.channels() == 2)
						temp.at<cv::Vec<double, 2>>(ii, jj) =
						img_org.at<cv::Vec<double, 2>>(i, j);
					else if (img_org.channels() == 1)
						temp.at<double>(ii, jj) = img_org.at<double>(i, j);
					else
						assert(0 && "error of image channels.");
				}
			}
		}
		return temp;
	}

	// take the real part of a complex img
	cv::Mat real(const cv::Mat img)
	{
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		return planes[0];
	}
	// take the image part of a complex img
	cv::Mat imag(const cv::Mat img)
	{
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		return planes[1];
	}
	// calculate the magnitde of a complex img
	cv::Mat magnitude(const cv::Mat img)
	{
		cv::Mat res;
		std::vector<cv::Mat> planes;
		cv::split(img, planes);
		if (planes.size() == 1)
			res = cv::abs(img);
		else if (planes.size() == 2)
			cv::magnitude(planes[0], planes[1], res);
		else
			assert(0 && "error: img size error!");
		return res;
	}
	// complex element-wise multiplication for 32Float type
	cv::Mat complexDotMultiplication(const cv::Mat &a, const cv::Mat &b)
	{
		cv::Mat res;

		res = complexDotMultiplicationCPU(a, b);

		return res;
	}

	cv::Mat complexDotMultiplicationCPU(const cv::Mat &a, const cv::Mat &b)
	{
		cv::Mat temp_a;
		cv::Mat temp_b;
		a.copyTo(temp_a);
		b.copyTo(temp_b);

		if (a.channels() == 1) // for single channel image a
		{
			std::vector<cv::Mat> a_vector =
			{ a, cv::Mat::zeros(a.size(), CV_32FC1) };
			cv::merge(a_vector, temp_a);
		}
		if (b.channels() == 1) // for single channel image b
		{
			std::vector<cv::Mat> b_vector =
			{ b, cv::Mat::zeros(b.size(), CV_32FC1) };
			cv::merge(b_vector, temp_b);
		}

		cv::Mat res = cv::Mat::zeros(temp_a.size(), CV_32FC2);
		//(a0+ia1)x(b0+ib1)=(a0b0-a1b1)+i(a0b1+a1b0)
		//#pragma omp parallel for collapse(2)
		for (int j = 0; j < temp_a.cols; j++)
		{
			for (int i = 0; i < temp_a.rows; i++)
			{
				res.at<cv::Vec2f>(i, j)[0] = temp_a.at<cv::Vec2f>(i, j)[0] * temp_b.at<cv::Vec2f>(i, j)[0] - temp_a.at<cv::Vec2f>(i, j)[1] * temp_b.at<cv::Vec2f>(i, j)[1];
				res.at<cv::Vec2f>(i, j)[1] = temp_a.at<cv::Vec2f>(i, j)[0] * temp_b.at<cv::Vec2f>(i, j)[1] + temp_a.at<cv::Vec2f>(i, j)[1] * temp_b.at<cv::Vec2f>(i, j)[0];
			}
		}
		return res;
	}
	
	// complex element-wise division
	cv::Mat complexDotDivision(const cv::Mat a, const cv::Mat b)
	{
		std::vector<cv::Mat> pa;
		std::vector<cv::Mat> pb;
		cv::split(a, pa);
		cv::split(b, pb);
		// Opencv if divide by ZERO, result is ZERO!
		cv::Mat divisor = 1. / (pb[0].mul(pb[0]) + pb[1].mul(pb[1]));
		//(a0+ia1)/(b0+ib1)=[(a0b0+a1b1)+i(a1b0-a0b1)] / divisor
		std::vector<cv::Mat> pres;
		pres.push_back((pa[0].mul(pb[0]) + pa[1].mul(pb[1])).mul(divisor));
		pres.push_back((pa[1].mul(pb[0]) - pa[0].mul(pb[1])).mul(divisor));

		cv::Mat res;
		cv::merge(pres, res);
		return res;
	}
	// the mulitiplciation of two complex matrix
	cv::Mat complexMatrixMultiplication(const cv::Mat &a, const cv::Mat &b)
	{
		if (a.empty() || b.empty())
			return a;

		if (a.cols != b.rows)
			assert(0 && "error: a and b size unmatched!");

		cv::Mat res(a.rows, b.cols, CV_32FC2);
		for (size_t i = 0; i < (size_t)res.rows; i++)
		{
			for (size_t j = 0; j < (size_t)res.cols; j++)
			{
				cv::Complex<float> rest(0, 0);
				for (size_t k = 0; k < (size_t)a.cols; k++)
				{
					rest += cv::Complex<float>(a.at<cv::Vec<float, 2>>(i, k)[0],
						a.at<cv::Vec<float, 2>>(i, k)[1]) *
						cv::Complex<float>(b.at<cv::Vec<float, 2>>(k, j)[0],
							b.at<cv::Vec<float, 2>>(k, j)[1]);
				}
				res.at<cv::Vec<float, 2>>(i, j) =
					cv::Vec<float, 2>(rest.re, rest.im);
			}
		}
		return res;
	}
	// impliment matlab c = convn(a,b) and convn(a, b, 'valid')
	cv::Mat complexConvolution(const cv::Mat a_input,
		const cv::Mat b_input,
		const bool valid)
	{
		cv::Mat res;
		cv::Mat a_temp, a, b;

		if (a_input.channels() == 1)
		{
			a_temp = real2complex(a_input);
		}
		else if (a_input.channels() == 2)
		{
			a_temp = a_input;
		}
		else if (a_input.channels() > 2)
		{
			assert(0 && "error: a_input's channel dimensions error!");
		}

		if (b_input.channels() == 1)
		{
			b = real2complex(b_input);
		}
		else if (b_input.channels() == 2)
		{
			b = b_input;
		}
		else if (b_input.channels() > 2)
		{
			assert(0 && "error: b_input's channel dimensions error!");
		}
		// padding with zeros
		a = cv::Mat::zeros(a_input.rows + b_input.rows - 1,
			a_input.cols + b_input.cols - 1,
			CV_32FC2);
		cv::Point pos(b_input.cols / 2, b_input.rows / 2);
		// copy to coresoponding location of the matrix
		a_temp.copyTo(a(cv::Rect(b_input.cols - 1 - pos.x,
			b_input.rows - 1 - pos.y,
			a_input.cols,
			a_input.rows)));

		rot90(b, 3); // flip around x and y axis

		std::vector<cv::Mat> va, vb;
		cv::split(a, va);
		cv::split(b, vb);

		cv::Mat r, i, r1, r2, i1, i2;
		cv::filter2D(va[0], r1, -1, vb[0],
			cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);
		cv::filter2D(va[1], r2, -1, vb[1],
			cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);
		cv::filter2D(va[0], i1, -1, vb[1],
			cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);
		cv::filter2D(va[1], i2, -1, vb[0],
			cv::Point(-1, -1), 0, cv::BORDER_ISOLATED);

		//(a0+ia1)x(b0+ib1)=(a0b0-a1b1)+i(a0b1+a1b0)
		r = r1 - r2; // a0b0-a1b1
		i = i1 + i2; // a0b1+a1b0

		cv::merge(std::vector<cv::Mat>({ r, i }), res);

		if (valid)
		{
			if (b_input.cols > a_input.cols || b_input.rows > a_input.rows)
			{
				return cv::Mat::zeros(0, 0, CV_32FC2);
			}
			else
			{
				return res(cv::Rect(b_input.cols - 1,
					b_input.rows - 1,
					a_input.cols - b_input.cols + 1,
					a_input.rows - b_input.rows + 1));
			}
		}
		else
		{
			return res;
		}
	}
	// change real mat to complex mat
	cv::Mat real2complex(const cv::Mat &x)
	{
		if (x.empty() || x.channels() == 2)
			return x;
		std::vector<cv::Mat> c = { x, cv::Mat::zeros(x.size(), CV_32FC1) };
		cv::Mat res;
		cv::merge(c, res);
		return res;
	}
	// mat conjugation
	cv::Mat mat_conj(const cv::Mat &org)
	{
		if (org.empty())
			return org;
		std::vector<cv::Mat_<float>> planes;
		cv::split(org, planes);
		planes[1] = -planes[1];
		cv::Mat result;
		cv::merge(planes, result);
		return result;
	}
	// sum up all the mat elements, just for float type.
	float mat_sum_f(const cv::Mat &org)
	{
		if (org.empty())
			return 0;
		float sum = 0;
		for (size_t r = 0; r < (size_t)org.rows; r++)
		{
			const float *orgPtr = org.ptr<float>(r);
			for (size_t c = 0; c < (size_t)org.cols; c++)
			{
				sum += orgPtr[c];
			}
		}
		return sum;
	}
	// double type version of mat_sum
	double mat_sum_d(const cv::Mat &org)
	{
		if (org.empty())
			return 0;
		double sum = 0;
		for (size_t r = 0; r < (size_t)org.rows; r++)
		{
			const double *orgPtr = org.ptr<double>(r);
			for (size_t c = 0; c < (size_t)org.cols; c++)
			{
				sum += orgPtr[c];
			}
		}
		return sum;
	}

} // namespace eco
