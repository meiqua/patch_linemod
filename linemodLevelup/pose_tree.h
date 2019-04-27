#pragma once

#include "linemodLevelup.h"

linemodLevelup::Pose_structure hinter_sampling(int level, float radius,
                                               float azimuth_range_min, float azimuth_range_max,
                                               float elev_range_min, float elev_range_max,
                                               float tilt_range_min, float tilt_range_max,
                                               float tilt_step);
