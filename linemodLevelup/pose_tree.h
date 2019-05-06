#pragma once

#include "linemodLevelup.h"

linemodLevelup::Pose_structure hinter_sampling(int level, float radius,
                                               float azimuth_range_min = 0, float azimuth_range_max = 2*CV_PI,
                                               float elev_range_min = -0.5*CV_PI, float elev_range_max = 0.5*CV_PI,
                                               float tilt_range_min = -CV_PI, float tilt_range_max = CV_PI,
                                               float tilt_step = CV_PI/10);
