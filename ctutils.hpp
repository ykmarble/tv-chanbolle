#ifndef CTUTILS_H
#define CTUTILS_H

#define EIGEN_NO_DEBUG

#include <eigen3/Eigen/Core>

namespace ctutils {

void normalize_image(Eigen::MatrixXf *img);

void normalize_image(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> *img);

Eigen::MatrixXf load_rawimage(const char *path);

void save_rawimage(const char *path, const Eigen::MatrixXf &img);

void show_image(const Eigen::MatrixXf &img);

void projection(const Eigen::MatrixXf &img ,Eigen::MatrixXf *proj);

void inv_projection(const Eigen::MatrixXf &proj, Eigen::MatrixXf *img);

}

#undef EIGEN_NO_DEBUG

#endif /* CTUTILS_H */
