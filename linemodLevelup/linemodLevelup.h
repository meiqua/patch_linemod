#ifndef CXXLINEMOD_H
#define CXXLINEMOD_H
#include <opencv2/core/core.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "pose_renderer.h"

class poseRefine{
public:
    poseRefine(): fitness(-1), inlier_rmse(-1){}
    void process(cv::Mat& sceneDepth, cv::Mat& modelDepth, cv::Mat& sceneK, cv::Mat& modelK,
                 cv::Mat& modelR, cv::Mat& modelT, int detectX, int detectY, double threshold = 0.007);

    cv::Mat get_depth_edge(cv::Mat& depth, int dilute_size = 5);

    void cannyTraceEdge(int rowOffset, int colOffset, int row, int col, cv::Mat& canny_edge, cv::Mat& mag_nms);

    cv::Mat result_refined;
    double fitness, inlier_rmse;
};

namespace linemodLevelup {

struct Feature {
    int x;
    int y;
    int label;
    int cluster;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;

    Feature() : x(0), y(0), label(0), cluster(0) {}
    Feature(int x, int y, int label);
};
inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct Template
{
    int width;
    int height;
    int tl_x;
    int tl_y;
    int pyramid_level;
    int clusters;
    std::vector<Feature> features;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;
};

class QuantizedPyramid
{
public:
    // Virtual destructor
    virtual ~QuantizedPyramid(){}

    virtual cv::Ptr<QuantizedPyramid> Clone(const cv::Mat& mask_crop, const cv::Rect& bbox) =0;

    /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
    virtual void quantize(cv::Mat& dst) const =0;
    /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
    virtual bool extractTemplate(Template& templ) const =0;

    /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
    virtual void pyrDown() =0;

    virtual void crop_by_mask(const cv::Mat& mask_crop, const cv::Rect& bbox) = 0;

protected:
    /// Candidate feature with a score
    struct Candidate
    {
        Candidate(int x, int y, int label, float score);

        /// Sort candidates with high score to the front
        bool operator<(const Candidate& rhs) const
        {
            return score > rhs.score;
        }

        Feature f;
        float score;
    };
    /**
   * \brief Choose candidate features so that they are not bunched together.
   *
   * \param[in]  candidates   Candidate features sorted by score.
   * \param[out] features     Destination vector of selected features.
   * \param[in]  num_features Number of candidates to select.
   * \param[in]  distance     Hint for desired distance between features.
   */
    static bool selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                        std::vector<Feature>& features,
                                        size_t num_features, float distance);
};

inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

class Modality
{
public:
    // Virtual destructor
    virtual ~Modality() {}

    /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
    cv::Ptr<QuantizedPyramid> process(const std::vector<cv::Mat> &src,
                                      const cv::Mat& mask = cv::Mat()) const
    {
        return processImpl(src, mask);
    }

    virtual std::string name() const =0;
    virtual void read(const cv::FileNode& fn) =0;
    virtual void write(cv::FileStorage& fs) const =0;

    /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
    static cv::Ptr<Modality> create(const std::string& modality_type);

protected:
    // Indirection is because process() has a default parameter.
    virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                                                  const cv::Mat& mask) const =0;
};

class ColorGradient : public Modality
{
public:

    ColorGradient();
    ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

    virtual std::string name() const;

    float weak_threshold;
    size_t num_features;
    float strong_threshold;
    virtual void read(const cv::FileNode& fn);
    virtual void write(cv::FileStorage& fs) const;
protected:
    virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                                                  const cv::Mat& mask) const;
};


class DepthNormal : public Modality
{
public:

    DepthNormal();

    DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
                int extract_threshold);

    virtual std::string name() const;

    int distance_threshold;
    int difference_threshold;
    size_t num_features;
    int extract_threshold;

    virtual void read(const cv::FileNode& fn);
    virtual void write(cv::FileStorage& fs) const;

protected:
    virtual cv::Ptr<QuantizedPyramid> processImpl(const std::vector<cv::Mat> &src,
                                                  const cv::Mat& mask) const;
};


struct Match
{
    Match()
    {
    }

    Match(int x, int y, float similarity, const std::string& class_id, int template_id);

    /// Sort matches with high similarity to the front
    bool operator<(const Match& rhs) const
    {
        // Secondarily sort on template_id for the sake of duplicate removal
        if (similarity != rhs.similarity)
            return similarity > rhs.similarity;
        else
            return template_id < rhs.template_id;
    }

    bool operator==(const Match& rhs) const
    {
        return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
    }

    int x;
    int y;
    float similarity;
    std::string class_id;
    int template_id;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;
};

inline
Match::Match(int _x, int _y, float _similarity, const std::string& _class_id, int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{}

struct Pose_structure{
    std::vector<cv::Mat> Ts;
    struct Node{
        int id;
        std::vector<int> adjs;
    };
    std::vector<Node> nodes;

    int select_behalf(std::vector<int>& current_cluster){
        return current_cluster[0]; // may select better behalf if use SO3 pose sampler
    }
};
class Detector
{
public:
    Detector();

    Detector(std::vector<int> T, int clusters_ = 16);
    Detector(int num_features, std::vector<int> T, int clusters_ = 16);

    Detector(const std::vector< cv::Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid);

    std::vector<Match> match(const std::vector<cv::Mat>& sources, float threshold, float active_ratio = 0.6,
                             const std::vector<std::string>& class_ids = std::vector<std::string>(),
                             const std::vector<int>& dep_anchors = std::vector<int>(), const int dep_range = 200,
                             const std::vector<cv::Mat>& masks = std::vector<cv::Mat>());

    std::vector<int> addTemplate(const std::vector<cv::Mat>& sources, const std::string& class_id,
                                 const cv::Mat object_mask = cv::Mat(),
                                 const std::vector<int> dep_anchors = std::vector<int>());

    const std::vector< cv::Ptr<Modality> >& getModalities() const { return modalities; }

    int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

    int pyramidLevels() const { return pyramid_levels; }

    const std::vector<Template>& getTemplates(const std::string& class_id, int template_id) const;

    int numTemplates() const;
//    int numTemplates(const std::string& class_id) const;
    int numClasses() const { return static_cast<int>(class_templates.size()); }

    std::vector<std::string> classIds() const;

    void read(const cv::FileNode& fn);
    void write(cv::FileStorage& fs) const;

    std::vector<Match> read_matches(std::string path);
    void write_matches(std::vector<Match> & matches, std::string path) const;

    std::string readClass(const cv::FileNode& fn, const std::string &class_id_override = "");
    void writeClass(const std::string& class_id, cv::FileStorage& fs) const;

    void readClasses(const std::vector<std::string>& class_ids,
                     const std::string& format = "templates_%s.yml.gz");
    void writeClasses(const std::string& format = "templates_%s.yml.gz") const;
    void clear_classes(){class_templates.clear();}

    struct TemplateStructure{
        std::vector<std::vector<Template>> templs;
        std::vector<std::vector<int>> templ_forest;
        int last_level_size;
    };
    TemplateStructure build_templ_structure(Pose_structure& structure, PoseRenderer& renderer);
    bool is_similar(cv::Mat& pose1, cv::Mat& pose2, int pyr_level, int stride, PoseRenderer& renderer);
    std::vector<Template> render_templ(cv::Mat& m4f, int level, PoseRenderer &renderer);
    std::map<std::string, TemplateStructure> class_templs_structure;

    // for test
    std::vector<cv::Vec3f> pts_test, pts_test2;
protected:

    int clusters;
    int num_features;
    std::vector< cv::Ptr<Modality> > modalities;
    int pyramid_levels;
    std::vector<int> T_at_level;

    typedef std::vector<Template> TemplatePyramid;
    typedef std::map<std::string, std::vector<TemplatePyramid> > TemplatesMap;
    TemplatesMap class_templates;

    typedef std::vector<cv::Mat> LinearMemories;
    // Indexed as [pyramid level][modality][quantized label]
    typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

    void matchClass(const LinearMemoryPyramid& lm_pyramid,
                    const std::vector<cv::Size>& sizes,
                    float threshold, float active_ratio, std::vector<Match>& matches,
                    const std::string& class_id,
                    const std::vector<TemplatePyramid>& template_pyramids) const;

    void matchClass_by_structure(const LinearMemoryPyramid& lm_pyramid,
                    const std::vector<cv::Size>& sizes,
                    float threshold, float active_ratio, std::vector<Match>& matches,
                    const std::string& class_id,
                    const TemplateStructure& template_structure) const;

};

cv::Ptr<linemodLevelup::Detector> getDefaultLINEMOD();
}

#endif
