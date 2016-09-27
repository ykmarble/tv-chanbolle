#include "ctutils.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>

namespace ctutils{

const int NumOfAngle = 512;

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
    char magic[4];
    unsigned int width, height;
    fread(magic, sizeof(char), 4, f);
    fread(&width, sizeof(unsigned int), 1, f);
    fread(&height, sizeof(unsigned int), 1, f);
    unique_ptr<float> img_seq(new float[width * height]);
    fread(img_seq.get(), sizeof(float), width * height, f);
    fclose(f);
    return Eigen::Map<Matrix<float, Dynamic, Dynamic, RowMajor>> (img_seq.get(), width, height);;
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

void inner_proj(MatrixXf *img, MatrixXf *proj, bool inverse) {
    /*
      `inverse`がfalseの時、`img`を`proj`に順投影する。
      `inverse`がtrueの時、`proj`を`img`に逆投影する。
      投影はpallarel beamジオメトリでpixel-drivenに行われる。
      各画素から垂線を下ろした先に検知器が存在しない、つまり`proj`の長さが足りない場合エラーになる。
      `proj`は高々画像の対角線+2の長さがあれば足りる。
     */
    // memo:
    // 画素間の幅を1、画像の中心を(0, 0)として座標系を取る。
    // 検知器間の幅も1とし、角度0の時の検知器の中心のy座標を0としてy軸に平行に検知器が並ぶとする
    // 平行ビームなので角度0の時のx座標は、画像の対角線の半分以上であれば何でもいい。

    // 各indexに履かせる負の下駄の大きさ
    float img_offset = (img->cols() - 1) / 2.0;
    float detector_offset = (proj->rows() - 1) / 2.0;
    float r = pow(img_offset, 2);

    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = (float)deg_i / NumOfAngle * 2 * M_PI;

        // 原点を通る検知器列に直行する直線 ax + by = 0
        float a = sin(deg);
        float b = -cos(deg);
        for (int x_i = 0; x_i < img->cols(); x_i++) {
            for (int y_i = 0; y_i < img->rows(); y_i++) {
                float x = x_i - img_offset;
                float y = img_offset - y_i;  // 行列表記と軸の方向が逆になることに注意

                // 円の外は除外
                //if (pow(x, 2) + pow(y, 2) > r)
                //    continue;

                // distは検知器中心からの距離を意味しているが、X線源に向かって右側を正とする
                // ローカル座標で表されている。
                int sign = a * x + b * y < 0 ? 1 : -1;  // 角度によらずこれで符号が出る
                float dist = sign * abs(a * x + b * y);

                // 影響のある検知器は高々2つ、検知器上のローカル座標系でより小さい方をlとする
                int l = floor(dist + detector_offset);
                int h = l + 1;
                float h_ratio = dist - floor(dist);
                float l_ratio = 1 - h_ratio;

                if (inverse) {
                    (*img)(y_i, x_i) += (*proj)(l, deg_i) * l_ratio;
                    (*img)(y_i, x_i) += (*proj)(h, deg_i) * h_ratio;
                } else {
                    float val = (*img)(y_i, x_i);
                    (*proj)(l, deg_i) += val * l_ratio;
                    (*proj)(h, deg_i) += val * h_ratio;
                }
            }
        }
    }
}

void inner_proj2(MatrixXf *img, MatrixXf *proj, bool inverse) {
    /*
      inner_proj の lay-driven な実装
     */
    VectorXf locus = VectorXf::Zero(img->cols());  // 整数値との交点
    for (int deg_i = 0; deg_i < NumOfAngle; deg_i++) {
        float deg = 2 * M_PI * deg_i / NumOfAngle;
        float lay_start;  // 始点
        float disp_lay;  // 変位 displacement
        if (deg < M_PI_4) {

        } else if (deg < 3 * M_PI_4) {

        } else if (deg < 5 * M_PI_4) {

        } else if (deg < 7 * M_PI_4) {

        } else {

        }

        for (int i = 0; i < img->cols(); i++) {

        }
    }
}

void linspace(float start, float step, VectorXf* out) {
    /*
      `start`から`step`ずつ値を増やして`out`に入れて返す
     */
    float y = start;
    for (int i = 0; i < out->size(); i++) {
        (*out)(i) = y;
        y += step;
    }
}

void projection(const MatrixXf &img ,MatrixXf *proj) {
    /*
      `img`に順投影を施し、`proj`に得られた値を加える。
      つまり、`proj`は呼び出し元で初期化されている必要がある。
     */
    //printf("projection:\n");
    inner_proj((MatrixXf*)&img, proj, false);
}

void inv_projection(const MatrixXf &proj, MatrixXf *img) {
    /*
      `proj`に逆投影を施し、`img`に得られた値を加える。
      つまり、`img`は呼び出し元で初期化されている必要がある。
     */
    //printf("inv_projection:\n");
    inner_proj(img, (MatrixXf*)&proj, true);
}

}
