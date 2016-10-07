#ifndef CTUTILS_H
#define CTUTILS_H

#define EIGEN_NO_DEBUG

#include <eigen3/Eigen/Core>

namespace ctutils {

Eigen::MatrixXf load_rawimage(const char *path);

void save_rawimage(const char *path, const Eigen::MatrixXf &img);

void show_image(const Eigen::MatrixXf &img);

}

#endif /* CTUTILS_H */
