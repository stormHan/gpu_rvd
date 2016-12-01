/*
 * Basic math operation in device.
 */

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

namespace Gpu_Rvd{

	__device__
	inline double distance2(double3 p1, double3 p2){
		return (p2.x - p1.x) * (p2.x - p1.x)
			+ (p2.y - p1.y) * (p2.y - p1.y)
			+ (p2.z - p1.z) * (p2.z - p1.z);
	}

	__device__
		inline double maxd(double d1, double d2)
	{
		return (d1 > d2) ? d1 : d2;
	}

	__device__
	inline	double3 add(double3 a, double3 b)
	{
		a.x = a.x + b.x;
		a.y = a.y + b.y;
		a.z = a.z + b.z;

		return a;
	}

	__device__
	inline	double3 sub(double3 a, double3 b)
	{
		a.x = a.x - b.x;
		a.y = a.y - b.y;
		a.z = a.z - b.z;

		return a;
	}

	__device__
	inline	double dot(double3 a, double3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__
	inline	int sgn(double _d)
	{
		return (_d > 0) ? 1 : (
			(_d < 0) ? -1 : 0
			);
	}

	__device__
	inline	double m_fabs(double x)
	{
		if (x > 0) return x;
		else return -x;
	}
}

#endif /* CUDA_MATH_H */