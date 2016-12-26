/*
	the implementation of geexapp
	*/

#include <graphics\geexapp.h>

namespace Geex{
	GeexApp* GeexApp::instance_ = nil;

	GeexApp::GeexApp(int argc, char** argv){
		//Only one instance at a time.
		instance_ = this;

		argc_ = argc;
		argv_ = argv;

		width_ = 800;
		height_ = 800;

	}

	GeexApp::~GeexApp() {
		instance_ = nil;
		//delete scene_;
	}

	void GeexApp::get_arg(std::string arg_name, GLint& arg_val){
		arg_name = std::string("-") + arg_name;
		for (int i = 1; i < argc_ - 1; ++i){
			if (argv_[i] == arg_name){
				arg_val = atoi(argv_[i + 1]);
			}
		}
	}

	void GeexApp::get_arg(std::string arg_name, GLfloat& arg_val){
		arg_name = std::string("-") + arg_name;
		for (int i = 0; i < argc_ - 1; ++i){
			if (argv_[i] == arg_name){
				arg_val = (float)atof(argv_[i + 1]);
			}
		}
	}

	void GeexApp::get_arg(std::string arg_name, GLboolean& arg_val) {
		std::string on_arg_name = std::string("+") + arg_name;
		std::string off_arg_name = std::string("-") + arg_name;
		for (int i = 1; i < argc_; i++) {
			if (argv_[i] == on_arg_name) {
				arg_val = GL_TRUE;
			}
			else if (argv_[i] == off_arg_name) {
				arg_val = GL_FALSE;
			}
		}
	}

	void GeexApp::get_arg(std::string arg_name, std::string& arg_val) {
		arg_name = std::string("-") + arg_name;
		for (int i = 1; i < argc_ - 1; i++) {
			if (argv_[i] == arg_name) {
				arg_val = argv_[i + 1];
			}
		}
	}

	std::string GeexApp::get_file_arg(std::string extension, int start_index){
		for (int i = start_index; i < argc_; ++i){
			std::string argument = argv_[i];
			int arg_size = argument.size();
			if (argument.substr(arg_size - 3, 3) == extension){
				return argument;
			}
		}
		return "";
	}
}