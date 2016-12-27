/*
	used to call OpenGl for showing the result
	
*/

#ifndef H_GRAPHICS_GEEXAPP
#define H_GRAPHICS_GEEXAPP

#include <graphics\geexob.h>
#include <basic\common.h>
#include <thid_party\glut_viewer\glut_viewer_gui.h>
#include <thid_party\GLsdk\gl.h>

namespace Geex{

	class GeexApp{

	public:
		GeexApp(int argc, char** argv);
		virtual ~GeexApp();

		int argc() { return argc_; }
		char** argv() { return argv_; }

		static GeexApp* instance() { return instance_; }
		Geexob* scene() { return scene_; }

		void main_loop();

		GLboolean in_main_loop() const { return in_main_loop_; }
		/*
		 * \brief used to identify the arguments to set specific values.
		 */
		void get_arg(std::string arg_name, GLint& arg_val);
		void get_arg(std::string arg_name, GLfloat& arg_val);
		void get_arg(std::string arg_name, GLboolean& arg_val);
		void get_arg(std::string arg_name, std::string& arg_val);

		/*
		 * \brief Gets the the filename with extension.
		 */
		std::string get_file_arg(std::string extension, int start_index = 1);

		virtual void init_scene();
		virtual void init_gui();

	protected:
		//Member-function versions of the callbacks
		virtual void initialize();
		virtual void display();
		virtual void overlay();
		virtual GLboolean mouse(float x, float y, int button, enum GlutViewerEvent event);

		// Callbacks
		static void initialize_CB();
		static void toggle_skybox_CB();
		static void display_CB();
		static void overlay_CB();
		static GLboolean mouse_CB(float x, float y, int button, enum GlutViewerEvent event);

	protected:
		//GeexApp management.
		static GeexApp* instance_;
		int argc_;
		char** argv_;
		int width_;
		int height_;

		GLboolean in_main_loop_;

		Geexob* scene_;
		
		//Rendering
		GLboolean hdr_;
		GLuint diffuse_cubemap_;
		GLuint cubemap_;
		GLboolean fp_filtering_;	//If Float point texture is supported.

		//GUI
		GlutViewerGUI::Container GUI_;
		GlutViewerGUI::ViewerProperties* viewer_properties_;
	};
}


#endif /* H_GRAPHICS_GEEXAPP */