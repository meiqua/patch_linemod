#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "np2mat/ndarray_converter.h"
#include "linemodLevelup.h"
#include "pose_renderer.h"
namespace py = pybind11;

PYBIND11_MODULE(patch_linemod_pybind, m) {
    NDArrayConverter::init_numpy();

    py::class_<poseRefine>(m, "poseRefine")
        .def(py::init<>())
        .def_readwrite("result_refined",&poseRefine::result_refined)
        .def_readwrite("inlier_rmse",&poseRefine::inlier_rmse)
        .def_readwrite("fitness",&poseRefine::fitness)
        .def("get_depth_edge", &poseRefine::get_depth_edge)
        .def("set_depth", &poseRefine::set_depth, py::arg("depth"), py::arg("K") = cv::Mat())
        .def("process", &poseRefine::process);

    py::class_<PoseRenderer>(m, "PoseRenderer")
            .def(py::init<std::string, cv::Mat, cv::Mat>(), py::arg("model_path"),
                 py::arg("depth") = cv::Mat(), py::arg("K") = cv::Mat())
            .def("view_dep", &PoseRenderer::view_dep)
            .def("set_K_width_height", &PoseRenderer::set_K_width_height)
            .def("render_depth", &PoseRenderer::render_depth, py::arg("init_poses"), py::arg("down_sample") = 1)
            .def("render_mask", &PoseRenderer::render_mask, py::arg("init_poses"), py::arg("down_sample") = 1)
            .def("render_depth_mask", &PoseRenderer::render_depth_mask, py::arg("init_poses"), py::arg("down_sample") = 1);

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
        .def("addTemplate", &linemodLevelup::Detector::addTemplate, py::arg("sources"), py::arg("class_id"),
             py::arg("object_mask") = cv::Mat(), py::arg("dep_anchors") = std::vector<int>())
        .def("writeClasses", &linemodLevelup::Detector::writeClasses)
        .def("clear_classes", &linemodLevelup::Detector::clear_classes)
        .def("readClasses", &linemodLevelup::Detector::readClasses)
        .def("write_matches", &linemodLevelup::Detector::write_matches)
        .def("read_matches", &linemodLevelup::Detector::read_matches)
        .def("match", &linemodLevelup::Detector::match, py::arg("sources"),
             py::arg("threshold"), py::arg("active_ratio"), py::arg("class_ids"),
             py::arg("dep_anchors"), py::arg("dep_range"), py::arg("masks")=cv::Mat())
        .def("getTemplates", &linemodLevelup::Detector::getTemplates)
            .def("numTemplates", &linemodLevelup::Detector::numTemplates);
}
