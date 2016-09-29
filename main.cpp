#include "ctutils.hpp"

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
    while (i < 1) {
        div_2(p_x, p_y, &div_p);
        preview = (*img) - div_p / lambda;
        grad(div_p - lambda * (*img), &grad_x, &grad_y);
        denom = 1 + tau * (grad_x.array().pow(2) + grad_y.array().pow(2)).sqrt();
        p_x += tau * grad_x;
        p_x = p_x.array() / denom.array();
        p_y += tau * grad_y;
        p_y = p_y.array() / denom.array();
        i++;
    }
    div_2(p_x, p_y, &div_p);
    *img = (*img) - div_p / lambda;
}

void tv_sirt(const MatrixXf &data, MatrixXf *img, double alpha, int n) {
    /*
      `data`をデータ項としてSIRT法を適用し再構成画像を得る。
      結果は`img`に格納される。
     */
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());;
    MatrixXf grad = MatrixXf::Zero(img->rows(), img->cols());;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%d(%d)\n", i, j);
            proj *= 0;
            ctutils::projection(*img, &proj);
            grad *= 0;
            ctutils::inv_projection(data - proj, &grad);
            *img += alpha * grad;
        }
        tv_chanbolle(img, alpha);
        ctutils::show_image(*img);
    }
}

}

int main(int argn, char** argv) {
    const double scale = 0.8;
    if (argn != 2) {
        printf("Usage: %s file\n", argv[0]);
        return 1;
    }
    MatrixXf img = ctutils::load_rawimage(argv[1]);
    const int size = ceil(img.rows() * scale);
    MatrixXf proj = MatrixXf::Zero(size, ctutils::NumOfAngle);
    ctutils::projection(img, &proj, size);
    img = MatrixXf::Zero(size, size);
    double alpha = 1 / ((double)img.rows() * img.cols() * 2);
    tv_sirt(proj, &img, alpha, 10);
    return 0;
}
