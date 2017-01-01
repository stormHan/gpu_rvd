/**
 * \file header/basic/counted.h
 * \brief Base class of reference-counted objects
 * to be used with smart pointers
 */

#ifndef H_BASIC_COUNTED
#define H_BASIC_COUNTED

#include <basic\common.h>

namespace Gpu_Rvd{

	/*
	 * \brief Base class for reference-counted objects
	 * \details Reference counted objects implement shared ownership
	 * with a simple mechanism: objects willing to share ownership on
	 * a Counted object must call ref() on this object and call unref()
	 * when they no longer need it. The object is destroyed when no more
	 * objects hold a reference on it( when the last holder calls unref()).
	 *
	 * Objects can benefit reference counted sharing simply by deriving 
	 * from Counted and implementing the virtual destructor.
	 *
	 * Reference acquisition and release can be done manually by explicitly
	 * calling ref() and unref() on the reference counted objects, or can be
	 * doen automatically by using SmartPointer<T>
	 * \see SmartPointer
	 */
	class Counted{
	public:
		/**
		 * \brief Increments the reference count
		 * \details This function must be called to share ownership
		 * on this object. Calling ref() will prevent this object from
		 * being deleted when someone else releases ownership.
		 */
		void ref() const{
			++nb_refs_;
		}

		/**
		* \brief Decrements the reference count
		* \details This function must be called to release ownership on this
		* object when it's no longer needed. Whwen the reference count
		* reaches the value of 0 (zero), the object is simply deleted.
		*/
		void unref() const {
			--nb_refs_;
			if (nb_refs_ == 0) {
				delete this;
			}
		}

		/**
		* \brief Check if the object is shared
		* \details An object is considered as shared if at least 2 client
		* objects have called ref() on this object.
		* \return \c true if the object is shared, \c false otherwise
		*/
		bool is_shared() const {
			return nb_refs_ > 1;
		}

		/**
		* \brief Increments the reference count
		* \details This calls ref() on object \p counted if it is not null.
		* \param[in] counted reference object to reference.
		*/
		static void ref(const Counted* counted) {
			if (counted != nil) {
				counted->ref();
			}
		}

		/**
		* \brief Decrements the reference count
		* \details This calls unref() on object \p counted if it is not null.
		* \param[in] counted reference object to dereference.
		*/
		static void unref(const Counted* counted) {
			if (counted != nil) {
				counted->unref();
			}
		}

	protected:
		/**
		* \brief Creates a reference counted object
		* \details This initializes the reference count to 0 (zero).
		*/
		Counted() :
			nb_refs_(0) {
		}

		/**
		* \brief Destroys a reference counted object
		* \details The destructor is never called directly but indirectly
		* through unref(). If the reference counter is not null when the
		* destructor is called the program dies with an assertion failure.
		*/
		virtual ~Counted();
	private:
		/* Forbid copy constructor */
		Counted(const Counted&);
		/* Forbid assignment operator */
		Counted& operator= (const Counted&);

		mutable int nb_refs_;
	};

}

#endif /* H_BASIC_COUNTED */
