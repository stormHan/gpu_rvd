/*
 * Basic math operation in device.
 */

#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cuda\cuda_common.h>

namespace Gpu_Rvd{

	__device__
	inline double distance2(const double3& p1, const double3& p2){
		return (p2.x - p1.x) * (p2.x - p1.x)
			+ (p2.y - p1.y) * (p2.y - p1.y)
			+ (p2.z - p1.z) * (p2.z - p1.z);
	}

	__device__
	inline double distance(const double3& p1, const double3& p2){
		return sqrt(distance2(p1, p2));
	}

	__device__
		inline double maxd(const double& d1, const double& d2)
	{
		return (d1 > d2) ? d1 : d2;
	}

	__device__
	inline	double3 add(const double3& a, const double3& b)
	{
		double3 d = { a.x + b.x,
			a.y + b.y,
			a.z + b.z };
		return d;
	}

	__device__
	inline	double3 sub(const double3& a, const double3& b)
	{
		double3 d = { a.x - b.x,
			a.y - b.y,
			a.z - b.z };
		return d;
	}

	__device__
	inline	double dot(const double3& a, const double3& b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	__device__
	inline	int sgn(const double& _d)
	{
		return (_d > 0) ? 1 : (
			(_d < 0) ? -1 : 0
			);
	}

	__device__
	inline	double m_fabs(const double& x)
	{
		if (x > 0) return x;
		else return -x;
	}

	__device__
		inline double3 cross(const double3& v1, const double3& v2){
		double3 d = {
			v1.y*v2.z - v1.z*v2.y,
			v1.z*v2.x - v1.x*v2.z,
			v1.x*v2.y - v1.y*v2.x
		};
		return d;
	}

	__device__
		inline double length(const double3& v){
		return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
	}
	__device__
	inline double computeTriangleArea(const double3& v1, const double3& v2, const double3& v3)
	{
		return 0.5 * length(cross(sub(v2, v1), sub(v3, v1)));
	}

	__device__
	inline void computeTriangleCentriod(
		const double3& p, const double3& q, const  double3& r, //double a, double b, double c,
		double3& Vg, double& V
		)
	{
		/*double abc = a + b + c;
		double area = computeTriangleArea(p, q, r);
		V = area / 3.0 * abc;

		double wp = a + abc;
		double wq = b + abc;
		double wr = c + abc;

		double s = area / 12.0;
		Vg.x = s * (wp * p.x + wq * q.x + wr * r.x);
		Vg.y = s * (wp * p.y + wq * q.y + wr * r.y);
		Vg.z = s * (wp * p.z + wq * q.z + wr * r.z);*/

		V = computeTriangleArea(p, q, r);
		Vg.x = (p.x + q.x + r.x) / 3.0 * V;
		Vg.y = (p.y + q.y + r.y) / 3.0 * V;
		Vg.z = (p.z + q.z + r.z) / 3.0 * V;
	}
}

#endif /* CUDA_MATH_H */