#include "ctutils.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

namespace {

using Eigen::MatrixXf;
using Eigen::Matrix;

void normalize_image(MatrixXf *img) {
    /*
      `img`を0から255の範囲に正規化
     */
    *img = ((*img).array() - (*img).minCoeff()).matrix();
    *img /= (*img).maxCoeff();
    *img *= 255;
}

void normalize_image(Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> *img) {
    /*
      `img`を0から255の範囲に正規化
     */
    *img = ((*img).array() - (*img).minCoeff()).matrix();
    *img /= (*img).maxCoeff();
    *img *= 255;
}

}

namespace ctutils{

using Eigen::MatrixXf;
using Eigen::Matrix;

MatrixXf load_rawimage(const char *path) {
    /*
      `path`から独自形式の画像を読み込む。画素値の正規化は行わない。
     */
    FILE* f = fopen(path, "rb");
    if (f == nullptr) {
        printf("failed to load image\n");
        return MatrixXf::Zero(-1, -1);
    }
    char magic[4];
    unsigned int width, height;
    fread(magic, sizeof(char), 4, f);
    fread(&width, sizeof(unsigned int), 1, f);
    fread(&height, sizeof(unsigned int), 1, f);
    std::unique_ptr<float> img_seq(new float[width * height]);
    fread(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
    return Eigen::Map<Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (img_seq.get(), height, width);
}

void save_rawimage(const char *path, const MatrixXf &img) {
    /*
      `path`に`img`の画像を独自形式で書き出す。書き出す前に画素値の正規化を行う。
     */
    char magic[] = {'P', '0', 0x00, 0x00};
    Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> img_t = img;
    normalize_image(&img_t);
    unsigned int width = img_t.cols();
    unsigned int height = img_t.rows();
    FILE* f = fopen(path, "wb");
    if (f == nullptr) {
        printf("failed to save image\n");
        return;
    }
    fwrite(magic, sizeof(char), 4, f);
    fwrite(&width, sizeof(unsigned int), 1, f);
    fwrite(&height, sizeof(unsigned int), 1, f);
    fwrite(&img_t(0), sizeof(float), width * height, f);
    fclose(f);
}

void show_image(const MatrixXf &img) {
    /*
      `img`をウィンドウに表示
     */
    MatrixXf normalized = img;
    normalize_image(&normalized);
    normalized /= 255;
    cv::Mat cvimg;
    cv::eigen2cv(normalized, cvimg);
    cv::imshow("img", cvimg);
    cv::waitKey();
}

}
