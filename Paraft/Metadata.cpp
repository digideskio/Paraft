#include "Metadata.h"

Metadata::Metadata(const string &fpath) {
    start_      = 0;
    end_        = 23;
    prefix_     = "";
    suffix_     = "raw";
    path_       = "/Users/Yang/Develop/ffv/sandbox/raw";
    tfPath_     = "/Users/Yang/Develop/ffv/sandbox/raw/config.tfe";
    timeFormat_ = "%03d";
    volumeDim_  = Vector3i(128, 64, 64);
}
