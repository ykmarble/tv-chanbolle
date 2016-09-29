#include "ctutils.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

namespace ctutils{

using namespace Eigen;
using namespace std;

void normalize_image(MatrixXf *img) {
    /*
      `img`を0から255の範囲に正規化
     */
    *img = ((*img).array() - (*img).minCoeff()).matrix();
    *img /= (*img).maxCoeff();
    *img *= 255;
}

void normalize_image(Matrix<float, Dynamic, Dynamic, RowMajor> *img) {
    /*
      `img`を0から255の範囲に正規化
     */
    *img = ((*img).array() - (*img).minCoeff()).matrix();
    *img /= (*img).maxCoeff();
    *img *= 255;
}

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
    unique_ptr<float> img_seq(new float[width * height]);
    fread(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
    return Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> (img_seq.get(), height, width);
}

void save_rawimage(const char *path, const MatrixXf &img) {
    /*
      `path`に`img`の画像を独自形式で書き出す。書き出す前に画素値の正規化を行う。
     */
    char magic[] = {'P', '0', 0x00, 0x00};
    Matrix<float, Dynamic, Dynamic, RowMajor> img_t = img;
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

void inner_proj(MatrixXf *img, MatrixXf *proj, bool inverse, float detector_length) {
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

    float img_offset = (img->cols() - 1) / 2.0;
    float detector_span = detector_length / proj->rows();
    float detector_offset = (proj->rows() - 1) / 2.0;

    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;

        // 原点を通る検知器列に直行する直線 ax + by = 0
        float a = sin(deg);
        float b = -cos(deg);

        float lay_width_2 = max(abs(a + b), abs(a - b)) / 2;
        for (int x_i = 0; x_i < img->cols(); x_i++) {
            for (int y_i = 0; y_i < img->rows(); y_i++) {
                float x = x_i - img_offset;
                float y = img_offset - y_i;  // 行列表記と軸の方向が逆になることに注意

                // distは検知器中心からの距離で、X線源に向かって左側を正とするローカル座標で表されている。
                float dist = a * x + b * y;

                float l_ratio = (dist - lay_width_2) / detector_span + detector_offset;
                float h_ratio = (dist + lay_width_2) / detector_span + detector_offset;
                int l = round(l_ratio);
                int h = round(h_ratio);
                l_ratio = 0.5 - (l_ratio - l);
                h_ratio = 0.5 + (h_ratio - h);

                // X線の両端が検知器列の内側に収まるようにする
                if (l > proj->rows() - 1 || h < 0)
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
                if (h > proj->rows() - 1) {
                    h = proj->rows() - 1;
                    if (l == h) {
                        l_ratio = h_ratio;
                        h_ratio = 0;
                    }
                    else
                        h_ratio = 1;
                }

                if (inverse) {
                    float val = 0;
                    val += (*proj)(l, deg_i) * l_ratio;
                    for (int i = l + 1; i < h; i++) {
                        val += (*proj)(i, deg_i);
                    }
                    val += (*proj)(h, deg_i) * h_ratio;
                    (*img)(y_i, x_i) += val / (2 * lay_width_2);;
                } else {
                    float val = (*img)(y_i, x_i) / (2 * lay_width_2);
                    (*proj)(l, deg_i) += val * l_ratio;
                    for (int i = l + 1; i < h; i++) {
                        (*proj)(i, deg_i) += val;
                    }
                    (*proj)(h, deg_i) += val * h_ratio;
                }
            }
        }
    }
}

void projection(const MatrixXf &img ,MatrixXf *proj) {
    /*
      `img`に順投影を施し、`proj`に得られた値を加える。
      つまり、`proj`は呼び出し元で初期化されている必要がある。
      検知器全体の長さは画像の幅に合わせられる｡
     */
    inner_proj((MatrixXf*)&img, proj, false, img.cols());
}

void projection(const MatrixXf &img ,MatrixXf *proj, float detector_length) {
    /*
      `img`に順投影を施し、`proj`に得られた値を加える。
      つまり、`proj`は呼び出し元で初期化されている必要がある。
      検知器全体の長さは`img`の画素間の幅を1とした時の`detector_length`となる｡
     */
    inner_proj((MatrixXf*)&img, proj, false, detector_length);
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    /*
      `proj`に逆投影を施し、`img`に得られた値を加える。
      つまり、`img`は呼び出し元で初期化されている必要がある。
      検知器全体の長さは画像の幅に合わせられる｡
     */
    inner_proj(img, (MatrixXf*)&proj, true, img->cols());
}

void inv_projection(const MatrixXf &proj, MatrixXf *img, float detector_length) {
    /*
      `proj`に逆投影を施し、`img`に得られた値を加える。
      つまり、`img`は呼び出し元で初期化されている必要がある。
      検知器全体の長さは`img`の画素間の幅を1とした時の`detector_length`となる｡
     */
    inner_proj(img, (MatrixXf*)&proj, true, detector_length);
}

}
