
#ifndef H_BASIC_VEC
#define H_BASIC_VEC

#include <basic\common.h>

namespace Gpu_Rvd{

	//--------------------------- VEC2 ------------------------------------------------------------------------------------------
	template<class T>
	struct vec2g{
		typedef vec2g<T> thisclass;

		vec2g() : x(0), y(0){}
		vec2g(T x_in, T y_in) : x(x_in), y(y_in){}

		inline T length2() const { return x*x + y*y; }
		inline T length() const { return sqrt(x*x + y*y); }

		// operators
		inline thisclass& operator+=(const thisclass& v) { x += v.x; y += v.y; return *this; }
		inline thisclass& operator-=(const thisclass& v) { x -= v.x; y -= v.y; return *this; }
		inline thisclass& operator*=(const thisclass& v) { x *= v.x; y *= v.y; return *this; }
		inline thisclass& operator/=(const thisclass& v) { x /= v.x; y /= v.y; return *this; }

		template <class T2>
		inline thisclass& operator *= (T2 s){
			x *= (T)s; y *= (T)s; return *this;
		}
		template <class T2>
		inline thisclass& operator /= (T2 s){
			x /= (T)s; y /= (T)s; return *this;
		}

		inline thisclass& operator+ (const thisclass v) { return thisclass(v.x + x, v.y + y); }
		inline thisclass& operator- (const thisclass v) { return thisclass(x - v.x, y - v.y); }
		template <class T2> inline thisclass operator* (T2 s) const { return thisclass(x*T(s), y*T(s)); }
		template <class T2> inline thisclass operator/ (T2 s) const { return thisclass(x / T(s), y / T(s)); }
		inline thisclass operator- () const { return thisclass(-x, -y); }

		inline T& operator[](int idx) {
			switch (idx) {
			case 0: return x; break;
			case 1: return y; break;
			}
			return x;
		}

		inline const T& operator[](int idx) const {
			switch (idx) {
			case 0: return x; break;
			case 1: return y; break;
			}
			return x;
		}

		T x;
		T y;
	};

	//--------------------------- VEC3 ------------------------------------------------------------------------------------------
	template<class T>
	struct vec3g{
		typedef vec3g thisclass;
		vec3g() : x(0), y(0), z(0){}
		vec3g(T x_in, T y_in, T z_in) : x(x_in), y(y_in), z(z_in) {}
		vec3g(const T* data){ x = data[0]; y = data[1]; z = data[2]; }

		inline T length2() const { return x*x + y*y + z*z; }
		inline T length() const { return sqrt(x*x + y*y + z*z); }

		// operators
		inline thisclass& operator+= (const thisclass& v){ x += v.x; y += v.y; z += v.z; return *this; }
		inline thisclass& operator-=(const thisclass& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
		inline thisclass& operator*=(const thisclass& v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
		inline thisclass& operator/=(const thisclass& v) { x /= v.x; y /= v.y; z /= v.z; return *this; }
		template <class T2> inline thisclass& operator*=(T2 s) { x *= T(s); y *= T(s); z *= T(s); return *this; }
		template <class T2> inline thisclass& operator/=(T2 s) { x /= T(s); y /= T(s); z /= T(s); return *this; }

		inline thisclass operator+ (const thisclass& v) const { return thisclass(x + v.x, y + v.y, z + v.z); }
		inline thisclass operator- (const thisclass& v) const { return thisclass(x - v.x, y - v.y, z - v.z); }
		template <class T2> inline thisclass operator* (T2 s) const { return thisclass(x*T(s), y*T(s), z*T(s)); }
		template <class T2> inline thisclass operator/ (T2 s) const { return thisclass(x / T(s), y / T(s), z / T(s)); }

		inline thisclass& operator- (){ return thisclass(-x, -y, -z); }

		inline T& operator[](int idx) {
			switch (idx) {
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default: ;
			}
			return x;
		}

		inline const T& operator[](int idx) const {
			switch (idx) {
			case 0: return x; break;
			case 1: return y; break;
			case 2: return z; break;
			default: ;
			}
			return x;
		}

		T x;
		T y;
		T z;
	};

	template<class T> 
	inline T dot(const vec3g<T>& v1, const vec3g<T>& v2) { return (T)(v1.x * v2.x + v1.y * v2.y + v1.z * v2.z); }

	template <class T>
	inline vec3g<T> cross(const vec3g<T>& v1, const vec3g<T>& v2){
		return vec3g<T>(
			v1.y*v2.z - v1.z*v2.y,
			v1.z*v2.x - v1.x*v2.z,
			v1.x*v2.y - v1.y*v2.x
			);
	}
	template <class T> inline vec3g<T> operator- (const vec3g<T>& v){
		return vec3g<T>(-v.x, -v.y, -v.z);
	}
	template <class T2, class T> inline vec3g<T> operator*(T2 s, const vec3g<T>& v) {
		return vec3g<T>(T(s)*v.x, T(s)*v.y, T(s)*v.z);
	}

	template <class T> inline vec3g<T> operator+(const vec3g<T>& v1, const vec3g<T>& v2) {
		return vec3g<T>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	template <class T> inline vec3g<T> operator-(const vec3g<T>& v1, const vec3g<T>& v2) {
		return vec3g<T>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	// Compatibility with GLSL
	template <class T> inline T length(const vec3g<T>& v) { return v.length(); }
	template <class T> inline T length2(const vec3g<T>& v) { return v.length2(); }
	template <class T> inline T distance2(const vec3g<T>& v1, const vec3g<T>& v2) { return length2(v2 - v1); }
	template <class T> inline T distance(const vec3g<T>& v1, const vec3g<T>& v2) { return length(v2 - v1); }
	template <class T> inline vec3g<T> normalize(const vec3g<T>& v) { return (T(1) / length(v)) * v; }

}

#endif /* H_BASIC_VEC*/