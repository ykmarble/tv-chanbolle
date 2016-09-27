#include "ctutils.hpp"
#include <eigen3/Eigen/Core>

namespace {

using namespace Eigen;

void grad(const MatrixXf mat, MatrixXf *out_x, MatrixXf *out_y) {
    int w = mat.cols();
    int h = mat.rows();
    *out_x = MatrixXf::Zero(w, h);
    *out_y = MatrixXf::Zero(w, h);
    (*out_x).leftCols(w - 1) = mat.rightCols(w - 1) - mat.leftCols(w - 1);
    (*out_y).topRows(w - 1) = mat.bottomRows(h - 1) - mat.topRows(h - 1);
}

void div_2(const MatrixXf &mat_x, const MatrixXf &mat_y, MatrixXf *out) {
    int w = mat_x.cols();
    int h = mat_x.rows();
    *out = MatrixXf::Zero(w, h);
    (*out).leftCols(w - 1) = mat_x.leftCols(w - 1);
    (*out).rightCols(w - 1) -= mat_x.leftCols(w - 1);
    (*out).topRows(h - 1) += mat_y.topRows(h - 1);
    (*out).bottomRows(h - 1) -= mat_y.topRows(h - 1);
}

void tv_chanbolle(MatrixXf *img, double lambda) {
    // p = (p + tau * grad(div(p) - lambda * f)) / (1 + tau * |grad(div(p)- lambda * f)|)
    MatrixXf p_x = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf p_y = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf div_p = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf grad_x = MatrixXf::Zero(img->cols(), img->rows());  // grad(div(p) - lambda * f)
    MatrixXf grad_y = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf denom = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf preview = MatrixXf::Zero(img->cols(), img->rows());
    double tau = 1.0 / 4;  // 1 / (2 * dimension)
    int i = 0;
    while (i < 30) {
        div_2(p_x, p_y, &div_p);
        preview = (*img) - div_p / lambda;
        ctutils::show_image(preview);
        grad(div_p - lambda * (*img), &grad_x, &grad_y);
        denom = 1 + tau * (grad_x.array().pow(2) + grad_y.array().pow(2)).sqrt();
        p_x += tau * grad_x;
        p_x = p_x.array() / denom.array();
        p_y += tau * grad_y;
        p_y = p_y.array() / denom.array();
        printf("%d\n", i);
        i++;
    }
    div_2(p_x, p_y, &div_p);
    *img = (*img) - div_p / lambda;
}

}

int main(int argn, char** argv) {
    if (argn != 2) {
        printf("Usage: %s file\n", argv[0]);
        return 1;
    }
    MatrixXf img = ctutils::load_rawimage(argv[1]);
    double lambda = 0.05;
    tv_chanbolle(&img, lambda);
    ctutils::show_image(img);
    return 0;
}
