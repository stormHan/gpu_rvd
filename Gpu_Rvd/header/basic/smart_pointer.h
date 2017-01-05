/**
 * \file header/basic/smart_pointer.h
 * \brief Pointer with automatic reference counting
 */

#ifndef H_BASIC_SMARTPOINTER
#define H_BASIC_SMARTPOINTER

#include <basic\common.h>

namespace Gpu_Rvd{
	
	/**
	 * \brief A smart pointer with reference-counted copy semantices
	 *
	 * \templateparam T the type of pointers stored in the SmartPointer.
	 * \details 
	 * SmartPointer has the ability of taking ownership of a pointer of
	 * type\p T and shares that ownership:once they take ownership, the group
     * of owners of a pointer become responsible for its deletion when the
     * last one of them releases that ownership.
     *
     * The object pointed to must implement the two following static
     * functions:
     * - T::ref(T*): to increment the reference count
     * - T::unref(T*): to decrement the reference count.
     *
     * More specifically, SmartPointer can be used with classes inheriting the
     * Counted class.
     * \see Counted
     */
	
	template <class T>
	class SmartPointer{
	public:
		/**
		 * \brief Creates an empty pointer
		 */
		SmartPointer() :
			pointer_(nil)
		{}

		/**
		* \brief Creates a smart pointer that owns a pointer
		* \details This calls T::ref() on the pointer \p ptr to take
		* ownership on it.
		* \param[in] ptr source pointer convertible to T
		*/
		SmartPointer(T* ptr) :
			pointer_(ptr){
			T::ref(pointer_);
		}

		/**
		* \brief Create a copy of a smart pointer
		* \details This calls T::ref() on the pointer help by \p rhs to take
		* ownership on it.
		* \param[in] rhs the smart pointer to copy
		*/
		SmartPointer(const SmartPointer<T>& rhs) :
			pointer_(rhs){
			T::ref(pointer_);
		}

		/**
		* \brief Assignment from a pointer
		* \details Releases ownership on the stored pointer as if reset()
		* were called and takes ownership on \p ptr.
		* \param[in] ptr a pointer convertible to T*
		* \return this smart pointer
		*/
		SmartPointer<T>& operator= (T* ptr){
			if (ptr != pointer_){
				T::unref(pointer_);
				pointer_ = ptr;
				T::ref(pointer_);
			}
			return *this;
		}

		SmartPointer<T>& operator= (const SmartPointer<T>& rhs){
			T* rhs_p = rhs.get();
			if (rhs_p != pointer_){
				T::unref(pointer_);
				pointer_ = rhs_p;
				T::ref(pointer_);
			}
			return *this;
		}

		/**
		* \brief Deletes a smart pointer
		* \details This calls T::unref() to release ownership on the help
		* pointer. If this smart pointer is the last one owning the pointer,
		* the pointer is deleted.
		*/
		~SmartPointer() {
			T::unref(pointer_);
		}

		/**
		* \brief Resets pointer
		* \details Releases ownership on the help pointer and resets it to
		* null. The smart pointer becomes as if it were default-constructed.
		* \note P.reset() is equivalent to assigning a nil pointer: p = nil
		*/
		void reset() {
			T::unref(pointer_);
			pointer_ = nil;
		}

		/**
		* \brief Dereferences object
		* \details Returns the stored pointer in order to dereference it.
		* This member function shall not be called if the stored pointer is a
		* null pointer.
		* \return the stored pointer if not null, or aborts otherwise.
		*/
		T& operator* () const {
			//if(pointer_ != nil);
				return *pointer_;
		}

		/**
		* \brief Dereferences object member
		* \details Returns the stored pointer in order to access one of its
		* members. This member function shall not be called if the stored
		* pointer is a null pointer.
		* \return the stored pointer if not null, or aborts otherwise.
		*/
		T* operator-> () const {
			//geo_assert(pointer_ != nil);
			return pointer_;
		}

		/*
		 * \brief Gets pointer
		 */
		T* get() const{
			return pointer_;
		}

		/**
		* \brief Conversion operator
		* \return the stored pointer
		*/
		operator T* () const {
			return pointer_;
		}

		/**
		 * \brief Checks if stored pointer is null
		 */
		bool is_nil(){
			return pointer_ == nil;

		}
	private:
		T* pointer_;
	};

	/**
	* \brief Equal operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is equal to the pointer
	* stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator== (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() == rhs.get();
	}

	/**
	* \brief Not equal operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is not equal to the
	* pointer stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator!= (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() != rhs.get();
	}

	/**
	* \brief Less than operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is less than the
	* pointer stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator< (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() < rhs.get();
	}

	/**
	* \brief Less or equal operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is less than or equal
	* to the pointer stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator<= (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() <= rhs.get();
	}

	/**
	* \brief Greater than operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is greater than the
	* pointer stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator> (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() > rhs.get();
	}

	/**
	* \brief Greater or equal operator
	* \param[in] lhs the first pointer to compare
	* \param[in] rhs the second pointer to compare
	* \return \c true if the pointer stored in \p lhs is greater than or
	* equal to the pointer stored in \p rhs.
	* \relates SmartPointer
	*/
	template <class T1, class T2>
	inline bool operator>= (const SmartPointer<T1>& lhs, const SmartPointer<T2>& rhs) {
		return lhs.get() >= rhs.get();
	}
}


#endif /* H_BASIC_SMARTPOINTER */