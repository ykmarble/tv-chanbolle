#ifndef PROJECTOR_H
#define PROJECTOR_H

#define EIGEN_NO_DEBUG

#include <eigen3/Eigen/Core>

namespace ctutils {

class projector {
    bool lay_driven;
    double detectors_len;

    template<bool inverse>
    void projection_lay(const Eigen::MatrixXf &rhs, Eigen::MatrixXf *lhs) const;

    template<bool inverse>
    void projection_pixel(const Eigen::MatrixXf &rhs, Eigen::MatrixXf *lhs) const;

public:
    projector(const double detectors_len);
    projector(const double detectors_len, const bool lay_driven);

    void set_detectors_length(const double length) {
        detectors_len = length;
    }

    void forward(const Eigen::MatrixXf &img, Eigen::MatrixXf *proj) const {
        if (lay_driven)
            projection_lay<false>(img, proj);
        else
            projection_pixel<false>(img, proj);
    }

    void backward(const Eigen::MatrixXf &proj, Eigen::MatrixXf *img) const {
        if (lay_driven)
            projection_lay<true>(proj, img);
        else
            projection_pixel<true>(proj, img);
    }
};

projector::projector(const double detectors_len):
    lay_driven(false) {
    set_detectors_length(detectors_len);
}

projector::projector(const double detectors_len, const bool lay_driven):
    lay_driven(lay_driven) {
    set_detectors_length(detectors_len);
}

template<bool inverse>
void projector::projection_lay(const Eigen::MatrixXf &rhs, Eigen::MatrixXf *lhs) const {

    *lhs *= 0;

    int NoX;
    int NoY;
    int NoDetector;
    int NoAngle;

    if (inverse) {
        NoX = lhs->cols();
        NoY = lhs->rows();
        NoDetector = rhs.rows();
        NoAngle = rhs.cols();
    } else {
        NoX = rhs.cols();
        NoY = rhs.rows();
        NoDetector = lhs->rows();
        NoAngle = lhs->cols();
    }

    const double dr     = detectors_len / (NoDetector - 1);
    const double dtheta = M_PI / NoAngle;
    const double img_offset = (NoX - 1) / 2.0;
    const double detector_offset = dr * (NoDetector - 1) / 2.0;

    for(int ti = 0; ti < NoAngle; ++ti) {
        const double th = ti * dtheta;
        const double sin_th = sin(th);
        const double cos_th = cos(th);
        const double abs_sin = std::abs(sin_th);
        const double abs_cos = std::abs(cos_th);

        if(abs_sin < abs_cos) {
            const double sin_cos = sin_th / cos_th;
            const double inv_cos_th = 1 / cos_th;
            const double inv_abs_cos = std::abs(inv_cos_th);

            for(int ri = 0; ri < NoDetector; ++ri) {
                const double r  = ri * dr - detector_offset;
                const double ray_offset = r * inv_cos_th;

                for(int xi = 0; xi < NoX; ++xi) {
                    const double xs   = xi - img_offset;
                    const double rayy = sin_cos * xs + ray_offset + img_offset;
                    const double yi   = std::floor(rayy);
                    const double aijp = rayy - yi;
                    const double aij  = 1 - aijp;

                    if(inverse) {
                        if (0 <= yi && yi < NoY)
                            (*lhs)(yi, xi) += aij * rhs(ri, ti) * inv_abs_cos;
                        if (0 <= yi+1 && yi+1 < NoY)
                            (*lhs)(yi+1, xi) += aijp * rhs(ri, ti) * inv_abs_cos;
                    } else {
                        if (0 <= yi && yi < NoY)
                            (*lhs)(ri, ti) += aij * rhs(yi, xi) * inv_abs_cos;
                        if (0 <= yi+1 && yi+1 < NoY)
                            (*lhs)(ri, ti) += aijp * rhs(yi+1, xi) * inv_abs_cos;
                    }
                }
            }
        } else {
            const double cos_sin = cos_th / sin_th;
            const double inv_sin_th = 1 / sin_th;
            const double inv_abs_sin = std::abs(inv_sin_th);

            for(int ri = 0; ri < NoDetector; ++ri) {
                const double r  = ri * dr - detector_offset;
                const double ray_offset = r * inv_sin_th;

                for(int yi = 0; yi < NoX ; ++yi) {
                    const double ys   = yi - img_offset;
                    const double rayx = cos_sin * ys - ray_offset + img_offset;
                    const double xi   = std::floor(rayx);
                    const double aijp = rayx - xi;
                    const double aij  = 1 - aijp;
                    if(inverse){
                        if (0 <= xi && xi < NoX)
                            (*lhs)(yi, xi) += aij * rhs(ri, ti) * inv_abs_sin;
                        if (0 <= xi+1 && xi+1 < NoX)
                            (*lhs)(yi, xi+1) += aijp * rhs(ri, ti) * inv_abs_sin;
                    } else {
                        if (0 <= xi && xi < NoX)
                            (*lhs)(ri, ti) += aij *rhs(yi, xi) * inv_abs_sin;
                        if (0 <= xi+1 && xi+1 < NoX)
                            (*lhs)(ri, ti) += aijp*rhs(yi, xi+1) * inv_abs_sin;
                    }
                }
            }
        }
    }
}


template<bool inverse>
void projector::projection_pixel(const Eigen::MatrixXf &rhs, Eigen::MatrixXf *lhs) const {
    /*
      `inverse`がfalseの時、`img`を`proj`に順投影する。
      `inverse`がtrueの時、`proj`を`img`に逆投影する。
      投影はpallarel beamジオメトリでpixel-drivenに行われる。
     */
    // memo:
    // 画素間の幅を1、画像の中心を(0, 0)として座標系を取る。
    // 検知器間の幅はdetector_lengthとprojの大きさから逆算｡
    // 角度0の時の検知器の中心のy座標を0としてy軸に平行に検知器が並ぶとする
    // 平行ビームなので角度0の時のx座標は、画像の対角線の半分以上であれば何でもいい。

    *lhs *= 0;

    int NoX;
    int NoY;
    int NoDetector;
    int NoAngle;

    if (inverse) {
        NoX = lhs->cols();
        NoY = lhs->rows();
        NoDetector = rhs.rows();
        NoAngle = rhs.cols();
    } else {
        NoX = rhs.cols();
        NoY = rhs.rows();
        NoDetector = lhs->rows();
        NoAngle = lhs->cols();
    }

    const double img_offset = (NoX - 1) / 2.0;
    const double dr = NoDetector / detectors_len;
    const double detector_offset = (NoDetector - 1) / 2.0;
    const double dth = M_PI / NoAngle;

    for (int deg_i = 0; deg_i < NoAngle; deg_i++) {
        const double deg = (double)deg_i * dth;

        // 原点を通る検知器列に直行する直線 ax + by = 0
        const double a = sin(deg);
        const double b = -cos(deg);

        //const double lay_width_2 = std::max(std::abs(a + b), std::abs(a - b)) / 2;
        for (int x_i = 0; x_i < NoX; x_i++) {
            for (int y_i = 0; y_i < NoY; y_i++) {
                const double x = x_i - img_offset;
                const double y = img_offset - y_i;  // 行列表記と軸の方向が逆になることに注意

                // distは検知器中心からの距離で、X線源に向かって左側を正とするローカル座標で表されている。
                const double dist = a * x + b * y;
                const double dist_on_det = dist * dr + detector_offset;
                const int l = std::floor(dist_on_det);
                const int h = l + 1;
                const double l_ratio = h - dist_on_det;
                const double h_ratio = 1 - l_ratio;

                if (inverse) {
                    float val = 0;
                    if (0 < l && l < NoDetector)
                        val += rhs(l, deg_i) * l_ratio;
                    if (0 < h && h < NoDetector)
                        val += rhs(h, deg_i) * h_ratio;
                    (*lhs)(y_i, x_i) += val;
                } else {
                    float val = rhs(y_i, x_i);
                    if (0 < l && l < NoDetector)
                        (*lhs)(l, deg_i) += val * l_ratio;
                    if (0 < h && h < NoDetector)
                        (*lhs)(h, deg_i) += val * h_ratio;
                }
            }
        }
    }
}

}

#endif /* PROJECTOR_H */
