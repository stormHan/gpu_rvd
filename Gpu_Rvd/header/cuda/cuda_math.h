/*
 * Basic math operation in device.
 */

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cuda\cuda_common.h>

namespace Gpu_Rvd{

	__device__
	inline double distance2(double3 p1, double3 p2){
		return (p2.x - p1.x) * (p2.x - p1.x)
			+ (p2.y - p1.y) * (p2.y - p1.y)
			+ (p2.z - p1.z) * (p2.z - p1.z);
	}

	__device__
	inline double distance(double3 p1, double3 p2){
		return sqrt(distance2(p1, p2));
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

	__device__
	inline double computeTriangleArea(double3 p1, double3 p2, double3 p3)
	{
		double a = distance(p1, p2);
		double b = distance(p2, p3);
		double c = distance(p1, p3);

		//Heron's Formula to compute the area of the triangle
		double p = (a + b + c) / 2;
		if (p >= a && p >= b && p >= c)
			return sqrt(p * (p - a) * (p - b) * (p - c));
		else
			return 0.0;
	}

	__device__
	inline void computeTriangleCentriod(
		double3 p, double3 q, double3 r, double a, double b, double c,
		double3& Vg, double& V
		)
	{
		double abc = a + b + c;
		double area = computeTriangleArea(p, q, r);
		V = area / 3.0 * abc;

		double wp = a + abc;
		double wq = b + abc;
		double wr = c + abc;

		double s = area / 12.0;
		Vg.x = s * (wp * p.x + wq * q.x + wr * r.x);
		Vg.y = s * (wp * p.y + wq * q.y + wr * r.y);
		Vg.z = s * (wp * p.z + wq * q.z + wr * r.z);
	}
}

#endif /* CUDA_MATH_H */