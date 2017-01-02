/**
 * \file header/basic/environment.cpp
 * \brief implementation of environment.h
 */

#include <basic\environment.h>
#include <basic\assert.h>
#include <basic\argused.h>

namespace {
	using namespace Gpu_Rvd;

	/**
	* \brief Root environment
	* \details The root environment stores properties as name-value pairs in
	* a dictionary.
	*/
	class RootEnvironment : public Environment {
	protected:
		/** \copydoc GEO::Environment::get_local_value() */
		virtual bool get_local_value(
			const std::string& name, std::string& value
			) const {
			ValueMap::const_iterator it = values_.find(name);
			if (it != values_.end()) {
				value = it->second;
				return true;
			}
			return false;
		}

		/** \copydoc GEO::Environment::set_local_value() */
		virtual bool set_local_value(
			const std::string& name, const std::string& value
			) {
			values_[name] = value;
			return true;
		}

		/** \brief ProcessEnvironment destructor */
		virtual ~RootEnvironment() {
		}

	private:
		/** \brief Stores the variable values by name */
		typedef std::map<std::string, std::string> ValueMap;
		ValueMap values_;
	};
}

namespace Gpu_Rvd{
	/************************************************************************/

	VariableObserver::VariableObserver(
		const std::string& var_name
		) :
		observed_variable_(var_name),
		environment_(nil)
	{
		environment_ = Environment::instance()->find_environment(var_name);
		geo_assert(environment_ != nil);
		environment_->add_observer(var_name, this);
	}

	VariableObserver::~VariableObserver() {
		environment_->remove_observer(observed_variable_, this);
	}

	/************************************************************************/

	void VariableObserverList::notify_observers(
		const std::string& value
		) {
		if (block_notify_) {
			return;
		}
		block_notify_ = true;
		for (size_t i = 0; i < observers_.size(); i++) {
			observers_[i]->value_changed(value);
		}
		block_notify_ = false;
	}

	void VariableObserverList::add_observer(
		VariableObserver* observer
		) {
		Observers::const_iterator it =
			std::find(observers_.begin(), observers_.end(), observer);
		geo_assert(it == observers_.end());
		observers_.push_back(observer);
	}

	void VariableObserverList::remove_observer(
		VariableObserver* observer
		) {
		Observers::iterator it =
			std::find(observers_.begin(), observers_.end(), observer);
		geo_assert(it != observers_.end());
		observers_.erase(it);
	}

	/************************************************************************/

	Environment::Environment_var Environment::instance_;

	Environment* Environment::instance() {
		if (instance_ == nil) {
			static bool created = false;
			if (created) {
				std::cerr
					<< "CRITICAL: Environment::instance() "
					<< "called after the instance was deleted"
					<< std::endl;
				geo_abort();
			}
			created = true;
			instance_ = new RootEnvironment();
			instance_->add_environment(new SystemEnvironment());
		}
		return instance_;
	}

	void Environment::terminate() {
		instance_.reset();
	}

	Environment::~Environment() {
	}

	bool Environment::add_environment(Environment* env) {
		environments_.push_back(env);
		return true;
	}

	bool Environment::has_value(const std::string& name) const {
		std::string value;
		return get_value(name, value);
	}

	bool Environment::set_value(
		const std::string& name, const std::string& value
		) {
		for (size_t i = 0; i < environments_.size(); i++) {
			if (environments_[i]->set_value(name, value)) {
				notify_local_observers(name, value);
				return true;
			}
		}
		if (set_local_value(name, value)) {
			notify_local_observers(name, value);
			return true;
		}
		return false;
	}

	bool Environment::get_value(
		const std::string& name, std::string& value
		) const {
		if (get_local_value(name, value)) {
			return true;
		}
		for (size_t i = 0; i < environments_.size(); i++) {
			if (environments_[i]->get_value(name, value)) {
				return true;
			}
		}
		return false;
	}

	std::string Environment::get_value(const std::string& name) const {
		std::string value;
		bool variable_exists = get_value(name, value);
		if (!variable_exists) {
			std::cerr << "Environment"
				<< "No such variable: " << name
				<< std::endl;
		}
		geo_assert(variable_exists);
		return value;
	}

	Environment* Environment::find_environment(const std::string& name) {
		std::string value;
		if (get_local_value(name, value)) {
			return this;
		}
		for (index_t i = 0; i<environments_.size(); ++i) {
			Environment* result = environments_[i]->find_environment(name);
			if (result != nil) {
				return result;
			}
		}
		return nil;
	}

	bool Environment::add_observer(
		const std::string& name, VariableObserver* observer
		) {
		observers_[name].add_observer(observer);
		return true;
	}

	bool Environment::remove_observer(
		const std::string& name, VariableObserver* observer
		) {
		ObserverMap::iterator obs = observers_.find(name);
		geo_assert(obs != observers_.end());
		obs->second.remove_observer(observer);
		return true;
	}

	bool Environment::notify_observers(
		const std::string& name, bool recursive
		) {
		std::string value = get_value(name);
		return notify_observers(name, value, recursive);
	}

	bool Environment::notify_observers(
		const std::string& name, const std::string& value,
		bool recursive
		) {
		if (recursive) {
			for (size_t i = 0; i < environments_.size(); i++) {
				environments_[i]->notify_observers(
					name, value, true
					);
			}
		}
		return notify_local_observers(name, value);
	}

	bool Environment::notify_local_observers(
		const std::string& name, const std::string& value
		) {
		ObserverMap::iterator it = observers_.find(name);
		if (it != observers_.end()) {
			it->second.notify_observers(value);
		}
		return true;
	}

	/************************************************************************/

	SystemEnvironment::~SystemEnvironment() {
	}

	bool SystemEnvironment::set_local_value(
		const std::string& name, const std::string& value
		) {
		geo_argused(name);
		geo_argused(value);
		return false;
	}

	bool SystemEnvironment::get_local_value(
		const std::string& name, std::string& value
		) const {
		// For the moment, deactivated under Windows
#ifdef GEO_OS_WINDOWS
		geo_argused(name);
		geo_argused(value);
		return false;
#else
		char* result = ::getenv(name.c_str());
		if (result != nil) {
			value = std::string(result);
		}
		return result != nil;
#endif
	}
}