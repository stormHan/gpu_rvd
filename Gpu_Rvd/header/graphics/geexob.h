

#ifndef H_GRAPHICS_GEEXOB
#define H_GRAPHICS_GEEXOB

#include <GLsdk\gl.h>

namespace Geex{

	class Geexob{
	public:

		Geexob();
		virtual ~Geexob();
		virtual void draw();

	protected:
		//GLhandleARB shader_;
	};
}

#endif /* H_GRAPHICS_GEEXOB */