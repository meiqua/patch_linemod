#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "linemodLevelup.h"
#include "pose_refine.h"

namespace py = pybind11;

PYBIND11_MODULE(patch_linemod_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<linemodLevelup::Match>(m,"Match")
            .def(py::init<>())
            .def_readwrite("x",&linemodLevelup::Match::x)
            .def_readwrite("y",&linemodLevelup::Match::y)
            .def_readwrite("similarity",&linemodLevelup::Match::similarity)
            .def_readwrite("class_id",&linemodLevelup::Match::class_id)
            .def_readwrite("template_id",&linemodLevelup::Match::template_id);

    py::class_<linemodLevelup::Template>(m,"Template")
            .def(py::init<>())
            .def_readwrite("width",&linemodLevelup::Template::width)
            .def_readwrite("height",&linemodLevelup::Template::height)
            .def_readwrite("pyramid_level",&linemodLevelup::Template::pyramid_level);


    py::class_<linemodLevelup::Detector>(m, "Detector")
        .def(py::init<>())
        .def(py::init<std::vector<int>, int>())
        .def(py::init<int, std::vector<int>, int>())
        .def("addTemplate", &linemodLevelup::Detector::addTemplate,
             py::arg("sources"),py::arg("class_id"),
             py::arg("object_mask")=cv::Mat(), py::arg("dep_anchors") = std::vector<int>())
        .def("writeClasses", &linemodLevelup::Detector::writeClasses)
        .def("clear_classes", &linemodLevelup::Detector::clear_classes)
        .def("write_matches", &linemodLevelup::Detector::write_matches)
        .def("read_matches", &linemodLevelup::Detector::read_matches)
        .def("readClasses", &linemodLevelup::Detector::readClasses)
        .def("match", &linemodLevelup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("active_ratio"), py::arg("class_ids"),
             py::arg("dep_anchors"), py::arg("dep_range"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &linemodLevelup::Detector::getTemplates)
            .def("numTemplates", &linemodLevelup::Detector::numTemplates);

    m.def("matches2poses", &poseRefine_adaptor::matches2poses,
          py::arg("matches"), py::arg("detector"), py::arg("saved_poses"),
          py::arg("K")=cv::Mat(), py::arg("top100")=100);


    py::class_<Mat4x4f>(m, "Mat4x4f", py::buffer_protocol())
       .def_buffer([](Mat4x4f &m) -> py::buffer_info {
            return py::buffer_info(
                &m[0][0],                               /* Pointer to buffer */
                sizeof(float),                          /* Size of one scalar */
                py::format_descriptor<float>::format(), /* Python struct-style format descriptor */
                2,                                      /* Number of dimensions */
                { 4, 4 },                 /* Buffer dimensions */
                { sizeof(float) * 4,             /* Strides (in bytes) for each index */
                  sizeof(float) }
            );
        });

    py::class_<cuda_icp::RegistrationResult>(m,"RegistrationResult")
            .def(py::init<>())
            .def_readwrite("fitness_", &cuda_icp::RegistrationResult::fitness_)
            .def_readwrite("inlier_rmse_", &cuda_icp::RegistrationResult::inlier_rmse_)
            .def_readwrite("transformation_", &cuda_icp::RegistrationResult::transformation_);

    py::class_<PoseRefine>(m, "PoseRefine")
            .def(py::init<std::string, cv::Mat, cv::Mat>(), py::arg("model_path"),
                 py::arg("depth") = cv::Mat(), py::arg("K") = cv::Mat())
            .def("view_dep", &PoseRefine::view_dep)
            .def("set_depth", &PoseRefine::set_depth)
            .def("set_K", &PoseRefine::set_K)
            .def("set_K_width_height", &PoseRefine::set_K_width_height)
            .def("set_max_dist_diff", &PoseRefine::set_max_dist_diff)
            .def("render_depth", &PoseRefine::render_depth, py::arg("init_poses"), py::arg("down_sample") = 1)
            .def("render_mask", &PoseRefine::render_mask, py::arg("init_poses"), py::arg("down_sample") = 1)
            .def("render_depth_mask", &PoseRefine::render_depth_mask, py::arg("init_poses"), py::arg("down_sample") = 1)
            .def("process_batch", &PoseRefine::process_batch, py::arg("init_poses"),py::arg("down_sample") = 2)
            .def("poses_extend", &PoseRefine::poses_extend, py::arg("init_poses"),
                  py::arg("degree_var") = CV_PI/10)
            .def("results_filter", &PoseRefine::results_filter, py::arg("detector"),
                  py::arg("rgb"), py::arg("results"), py::arg("active_thresh") = 70,
                  py::arg("fitness_thresh") = 0.6f, py::arg("rmse_thresh") = 0.005f);
}
