#include "ctutils.hpp"
#include "projector.hpp"

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

double total_variation(const MatrixXf &img) {
    MatrixXf grad_x;
    MatrixXf grad_y;
    grad(img, &grad_x, &grad_y);
    return (grad_x.array().pow(2) + grad_y.array().pow(2)).sqrt().sum();
}

double mse(const MatrixXf &x, const MatrixXf &x_true) {
    return (x - x_true).array().pow(2).sum() / (x.cols() * x.rows());
}

void tv_chanbolle(MatrixXf *img, const double lambda) {
    // p = (p + tau * grad(div(p) - lambda * f)) / (1 + tau * |grad(div(p)- lambda * f)|)
    MatrixXf p_x = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf p_y = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf div_p = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf grad_x = MatrixXf::Zero(img->cols(), img->rows());  // grad(div(p) - lambda * f)
    MatrixXf grad_y = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf denom = MatrixXf::Zero(img->cols(), img->rows());
    MatrixXf preview = MatrixXf::Zero(img->cols(), img->rows());
    const double tau = 1.0 / 4;  // 1 / (2 * dimension)
    const double size = img->rows() * img->cols();
    double last_diff = 0;
    double new_diff = 10;
    int i = 0;
    while (i < 100 && std::abs(new_diff - last_diff) > 0.001) {
        last_diff = new_diff;
        div_2(p_x, p_y, &div_p);
        preview = (*img) - div_p * lambda;
        grad(div_p - (*img) / lambda, &grad_x, &grad_y);
        denom = 1 + tau * (grad_x.array().pow(2) + grad_y.array().pow(2)).sqrt();
        p_x += tau * grad_x;
        p_x = p_x.array() / denom.array();
        p_y += tau * grad_y;
        p_y = p_y.array() / denom.array();
        new_diff =  div_p.array().abs().sum() * lambda / size;
        i++;
    }
    div_2(p_x, p_y, &div_p);
    *img -=  div_p * lambda;
}

void sirt(const ctutils::projector A, const MatrixXf &data, MatrixXf *img, const double alpha, const int n) {
    /*
      `data`をデータ項としてSIRT法を適用し再構成画像を得る。
      結果は`img`に格納される。
     */
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf grad = MatrixXf::Zero(img->rows(), img->cols());
    for (int i = 0; i < n; i++) {
        A.forward(*img, &proj);
        printf("%f\n", mse(proj, data));
        A.backward(data - proj, &grad);
        *img += alpha * grad;
        ctutils::show_image(*img);
    }
}

void proximal_gradient(const ctutils::projector A, const MatrixXf &data, MatrixXf *img, const double alpha, const int n) {
    MatrixXf w_pre = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf w = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf grad = MatrixXf::Zero(img->rows(), img->cols());
    double s_pre;
    double s = 1;
    for (int i = 0; i < n; i++) {
        w_pre = w;
        A.forward(*img, &proj);
        A.backward(proj - data, &grad);
        w = *img - alpha * grad;
        //tv_chanbolle(&w, alpha);
        //s_pre = s;
        //s = (1 + std::sqrt(1 + 4 * s * s)) / 2;
        *img =  w ;//+ (s_pre - 1 / s) * (w - w_pre);

        A.forward(*img, &proj);
        ctutils::show_image(*img);
        printf("%f\n", /*total_variation(*img) +*/ mse(proj, data));
    }
}

void app(const ctutils::projector A, const MatrixXf &data, MatrixXf *img, const double sigma, const double tau) {
    MatrixXf mu = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf mu_bar = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf proj = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf mu_img = MatrixXf::Zero(img->rows(), img->cols());
    for (int i = 0; i < 500; i++) {
        proj *= 0;
        A.forward(*img, &proj);
        mu_bar += sigma * (proj - data - mu);
        mu_img *= 0;
        A.backward(mu_bar, &mu_img);
        *img -= tau * mu_img;
        //tv_chanbolle(img, tau);
        proj *= 0;
        A.forward(*img, &proj);
        mu += sigma * (proj - data - mu);
        printf("%f\n", mse(proj, data) / 2);
    }
}

void cp(const ctutils::projector A, const MatrixXf &data, MatrixXf *img,
        const double sigma, const double tau, const double lambda) {
    MatrixXf p = MatrixXf::Zero(data.rows(), data.cols());      // Ax - b < e
    MatrixXf q_x = MatrixXf::Zero(img->rows(), img->cols());    // Dx
    MatrixXf q_y = MatrixXf::Zero(img->rows(), img->cols());    // Dy
    MatrixXf u = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf pre_u = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf u_grad_x = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf u_grad_y = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf u_bar = MatrixXf::Zero(img->rows(), img->cols());  // x
    MatrixXf tmpproj = MatrixXf::Zero(data.rows(), data.cols());
    MatrixXf tmpimg = MatrixXf::Zero(img->rows(), img->cols());
    MatrixXf divq = MatrixXf::Zero(img->rows(), img->cols());
    for(int i = 0; i < 500; i++){
        tmpproj *= 0;
        A.forward(u_bar, &tmpproj);
        p = p + sigma * (tmpproj - data) / (1 + sigma);

        grad(u_bar, &u_grad_x, &u_grad_y);
        q_x += sigma * u_grad_x;
        q_y += sigma * u_grad_y;
        q_x = q_x.array().max(lambda);
        q_x = q_x.array().min(-lambda);
        q_y = q_y.array().max(lambda);
        q_y = q_y.array().min(-lambda);

        tmpimg *= 0;
        A.backward(p, &tmpimg);
        div_2(q_x, q_y, &divq);
        pre_u = u;
        u += tau * divq - tau * tmpimg;

        u_bar = 2 * u - pre_u;

        tmpproj *= 0;
        A.forward(u, &tmpproj);
        printf("%f\n", lambda * total_variation(u) + mse(tmpproj, data) / 2);
    }
}

double calc_norm_AtA(const ctutils::projector &A, const int img_w, const int det_w, const int angles) {
    MatrixXf x = MatrixXf::Constant(img_w, img_w, 256);
    MatrixXf img = MatrixXf::Zero(img_w, img_w);
    MatrixXf proj = MatrixXf::Zero(det_w, angles);
    for (int i = 0; i < 5; i++) {
        A.forward(x, &proj);
        A.backward(proj, &img);
        A.forward(img, &proj);
        A.backward(proj, &img);

        x = img / img.norm();

        A.forward(x, &proj);
        A.backward(proj, &img);

        printf("%f\n", img.norm());
    }

    return img.norm();
}

double calc_norm_AtA(const ctutils::projector &A, const MatrixXf &img, const MatrixXf &proj) {
    return calc_norm_AtA(A, img.cols(), proj.rows(), proj.cols());
}

double calc_norm_ANabla(const ctutils::projector A, const int img_w, const int det_w, const int angles) {
    MatrixXf x = MatrixXf::Constant(img_w, img_w, 256);
    MatrixXf img = MatrixXf::Zero(img_w, img_w);
    MatrixXf grad_x = MatrixXf::Zero(img_w, img_w);
    MatrixXf grad_y = MatrixXf::Zero(img_w, img_w);
    MatrixXf div = MatrixXf::Zero(img_w, img_w);
    MatrixXf proj = MatrixXf::Zero(det_w, angles);
    for (int i = 0; i < 3; i++) {
        A.forward(x, &proj);
        A.backward(proj, &img);

        grad(x, &grad_x, &grad_y);
        div_2(grad_x, grad_y, &div);

        x = img - div;
        x /= x.norm();

        A.forward(x, &proj);
        grad(x, &grad_x, &grad_y);

        printf("%f\n", std::sqrt(proj.squaredNorm() + grad_x.squaredNorm() + grad_y.squaredNorm()));
    }

    return std::sqrt(proj.squaredNorm() + grad_x.squaredNorm() + grad_y.squaredNorm());
}

double calc_norm_ANabla(const ctutils::projector &A, const MatrixXf &img, const MatrixXf &proj) {
    return calc_norm_AtA(A, img.cols(), proj.rows(), proj.cols());
}


}

int main(int argn, char** argv) {
    const int NumOfAngle = 220;
    const double scale = 1.;

    if (argn != 2) {
        printf("Usage: %s file\n", argv[0]);
        return 1;
    }
    MatrixXf img = ctutils::load_rawimage(argv[1]);
    const int detector_length = ceil(img.rows() * scale);
    MatrixXf proj = MatrixXf::Zero(detector_length, NumOfAngle);
    ctutils::projector A(detector_length, false);
    A.forward(img, &proj);
    A.backward(proj, &img);
    ctutils::show_image(proj);
    ctutils::show_image(img);
    const double alpha = 2. / (img.cols() * img.rows());
    printf("%f\n", alpha);
    A.forward(img, &proj);

    img *= 0;
    sirt(A, proj, &img, alpha, 100);
    ctutils::show_image(img);
    //proximal_gradient(A, proj, &img, alpha, 500);
    //cp(A, proj, &img, alpha, alpha, 0);
    //app(A, proj, &img, alpha, alpha);
    return 0;
}
