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

    const double dr     = detectors_len / NoDetector;
    const double dtheta = M_PI / NoAngle;
    const double img_offset = (NoX - 1) / 2.0;
    const double detector_offset = (NoDetector - 1) / 2.0;

    for(int ti = 0; ti < NoAngle; ++ti) {
        const double th = ti*dtheta;
        const double sin_th = sin(th);
        const double cos_th = cos(th);
        const double abs_sin = std::abs(sin_th);
        const double abs_cos = std::abs(cos_th);
        if(abs_sin < abs_cos) {
            for(int ri = 0; ri < NoDetector; ++ri) {
                const double r  = ri * dr - detector_offset;

                for(int xi = 0; xi < NoX; ++xi) {
                    const double xs   = xi - img_offset;
                    const double rayy = sin_th / cos_th * xs + r / cos_th;
                    const double aij  = (std::ceil(rayy) - rayy) / abs_cos;
                    const double aijp = (rayy - std::floor(rayy)) / abs_cos;
                    const double yi   = std::floor(rayy + detector_offset);
                    if(inverse) {
                        if (0 <= yi && yi < NoY)
                            (*lhs)(yi, xi) += aij * rhs(ri, ti);
                        if (0 <= yi+1 && yi+1 < NoY)
                            (*lhs)(yi+1, xi) += aijp * rhs(ri, ti);
                    } else {
                        if (0 <= yi && yi < NoY)
                            (*lhs)(ri, ti) += aij * rhs(yi, xi);
                        if (0 <= yi+1 && yi+1 < NoY)
                            (*lhs)(ri, ti) += aijp * rhs(yi+1, xi);
                    }
                }
            }
        } else {
            for(int ri = 0; ri < NoDetector; ++ri) {
                const double r  = ri * dr - detector_offset;

                for(int yi = 0; yi < NoX ; ++yi) {
                    const double ys   = yi - img_offset;
                    const double rayx = cos_th / sin_th * ys - r / sin_th;
                    const double aij  = (std::ceil(rayx) - rayx) / abs_sin;
                    const double aijp = (rayx - std::floor(rayx)) / abs_sin;
                    const double xi   = std::floor(rayx + detector_offset);
                    if(inverse){
                        if (0 <= xi && xi < NoX)
                            (*lhs)(yi, xi) += aij * rhs(ri, ti);
                        if (0 <= xi+1 && xi+1 < NoX)
                            (*lhs)(yi, xi+1) += aijp * rhs(ri, ti);
                    } else {
                        if (0 <= xi && xi < NoX)
                            (*lhs)(ri, ti) += aij *rhs(yi, xi);
                        if (0 <= xi+1 && xi+1 < NoX)
                            (*lhs)(ri, ti) += aijp*rhs(yi, xi+1);
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
    const double detector_span = detectors_len / NoDetector;
    const double detector_offset = (NoDetector - 1) / 2.0;

    for (int deg_i = 0; deg_i < NoAngle; deg_i++) {
        const double deg = (double)deg_i / NoAngle * M_PI;

        // 原点を通る検知器列に直行する直線 ax + by = 0
        const double a = sin(deg);
        const double b = -cos(deg);

        const double lay_width_2 = std::max(std::abs(a + b), std::abs(a - b)) / 2;
        for (int x_i = 0; x_i < NoX; x_i++) {
            for (int y_i = 0; y_i < NoY; y_i++) {
                const double x = x_i - img_offset;
                const double y = img_offset - y_i;  // 行列表記と軸の方向が逆になることに注意

                // distは検知器中心からの距離で、X線源に向かって左側を正とするローカル座標で表されている。
                const double dist = a * x + b * y;

                double l_ratio = (dist - lay_width_2) / detector_span + detector_offset;
                double h_ratio = (dist + lay_width_2) / detector_span + detector_offset;
                int l = round(l_ratio);
                int h = round(h_ratio);
                l_ratio = 0.5 - (l_ratio - l);
                h_ratio = 0.5 + (h_ratio - h);

                // X線の両端が検知器列の内側に収まるようにする
                if (l > NoDetector - 1 || h < 0)
                    continue;
                if (l < 0) {
                    l = 0;
                    if (l == h){
                        l_ratio = h_ratio;  // 逆端まで丸める場合､比が1に満たないので元の値を考慮する必要がある
                        h_ratio = 0;
                    }
                    else
                        l_ratio = 1;
                }
                if (h > NoDetector - 1) {
                    h = NoDetector - 1;
                    if (l == h) {
                        l_ratio = h_ratio;
                        h_ratio = 0;
                    }
                    else
                        h_ratio = 1;
                }

                if (inverse) {
                    float val = 0;
                    val += rhs(l, deg_i) * l_ratio;
                    for (int i = l + 1; i < h; i++) {
                        val += rhs(i, deg_i);
                    }
                    val += rhs(h, deg_i) * h_ratio;
                    (*lhs)(y_i, x_i) += val / (2 * lay_width_2);
                } else {
                    float val = rhs(y_i, x_i) / (2 * lay_width_2);
                    (*lhs)(l, deg_i) += val * l_ratio;
                    for (int i = l + 1; i < h; i++) {
                        (*lhs)(i, deg_i) += val;
                    }
                    (*lhs)(h, deg_i) += val * h_ratio;
                }
            }
        }
    }
}

}

#endif /* PROJECTOR_H */
