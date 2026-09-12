#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" {

int compute_dhash64(const uint8_t *bgr, int width, int height,
                    uint64_t *out_hash) {
  if (!bgr || width < 9 || height < 8 || !out_hash) {
    return -1;
  }

  // Target grid: 9 columns x 8 rows
  uint8_t grid[8][9];

  // Compute average intensity for each cell in 9x8 grid
  const double step_x = static_cast<double>(width) / 9.0;
  const double step_y = static_cast<double>(height) / 8.0;

  for (int r = 0; r < 8; ++r) {
    const int y_start = static_cast<int>(r * step_y);
    const int y_end = static_cast<int>((r + 1) * step_y);

    for (int c = 0; c < 9; ++c) {
      const int x_start = static_cast<int>(c * step_x);
      const int x_end = static_cast<int>((c + 1) * step_x);

      uint64_t sum_gray = 0;
      int count = 0;

      const int sample_step_y = std::max(1, (y_end - y_start) / 8);
      const int sample_step_x = std::max(1, (x_end - x_start) / 8);

      for (int y = y_start; y < y_end; y += sample_step_y) {
        const uint8_t *row = bgr + (y * width * 3);
        for (int x = x_start; x < x_end; x += sample_step_x) {
          const int idx = x * 3;
          uint32_t gray =
              (row[idx] * 29 + row[idx + 1] * 150 + row[idx + 2] * 77) >> 8;
          sum_gray += gray;
          count++;
        }
      }

      grid[r][c] = (count > 0) ? static_cast<uint8_t>(sum_gray / count) : 0;
    }
  }

  uint64_t hash = 0;
  int bit_idx = 0;
  for (int r = 0; r < 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      if (grid[r][c] > grid[r][c + 1]) {
        hash |= (static_cast<uint64_t>(1) << bit_idx);
      }
      bit_idx++;
    }
  }

  *out_hash = hash;
  return 0;
}

int detect_changed_roi(const uint8_t *frame1, const uint8_t *frame2, int width,
                       int height, int threshold, float *out_diff_ratio,
                       int *out_min_x, int *out_min_y, int *out_max_x,
                       int *out_max_y) {
  if (!frame1 || !frame2 || width <= 0 || height <= 0 || !out_diff_ratio) {
    return -1;
  }

  int min_x = width;
  int min_y = height;
  int max_x = -1;
  int max_y = -1;
  int64_t changed_count = 0;

  const int stride = width * 3;

  for (int y = 0; y < height; ++y) {
    const uint8_t *p1 = frame1 + y * stride;
    const uint8_t *p2 = frame2 + y * stride;

    for (int x = 0; x < width; ++x) {
      const int idx = x * 3;
      int diff_b =
          std::abs(static_cast<int>(p1[idx]) - static_cast<int>(p2[idx]));
      int diff_g = std::abs(static_cast<int>(p1[idx + 1]) -
                            static_cast<int>(p2[idx + 1]));
      int diff_r = std::abs(static_cast<int>(p1[idx + 2]) -
                            static_cast<int>(p2[idx + 2]));

      if (diff_b > threshold || diff_g > threshold || diff_r > threshold) {
        changed_count++;
        if (x < min_x)
          min_x = x;
        if (x > max_x)
          max_x = x;
        if (y < min_y)
          min_y = y;
        if (y > max_y)
          max_y = y;
      }
    }
  }

  const double total_pixels = static_cast<double>(width) * height;
  *out_diff_ratio = static_cast<float>(changed_count / total_pixels);

  if (out_min_x && out_min_y && out_max_x && out_max_y) {
    if (changed_count > 0) {
      *out_min_x = min_x;
      *out_min_y = min_y;
      *out_max_x = max_x;
      *out_max_y = max_y;
    } else {
      *out_min_x = 0;
      *out_min_y = 0;
      *out_max_x = 0;
      *out_max_y = 0;
    }
  }

  return 0;
}

int fast_downsample_bgr(const uint8_t *src, int src_w, int src_h, uint8_t *dst,
                        int dst_w, int dst_h) {
  if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
    return -1;
  }

  const double scale_x = static_cast<double>(src_w) / dst_w;
  const double scale_y = static_cast<double>(src_h) / dst_h;

  const int src_stride = src_w * 3;
  const int dst_stride = dst_w * 3;

  for (int dy = 0; dy < dst_h; ++dy) {
    const double src_y = dy * scale_y;
    const int sy = std::min(static_cast<int>(src_y), src_h - 1);
    const uint8_t *src_row = src + sy * src_stride;
    uint8_t *dst_row = dst + dy * dst_stride;

    for (int dx = 0; dx < dst_w; ++dx) {
      const double src_x = dx * scale_x;
      const int sx = std::min(static_cast<int>(src_x), src_w - 1);

      const int s_idx = sx * 3;
      const int d_idx = dx * 3;

      dst_row[d_idx] = src_row[s_idx];         // B
      dst_row[d_idx + 1] = src_row[s_idx + 1]; // G
      dst_row[d_idx + 2] = src_row[s_idx + 2]; // R
    }
  }

  return 0;
}

int fast_draw_box(uint8_t *bgr, int width, int height, int x, int y, int box_w,
                  int box_h, uint8_t b, uint8_t g, uint8_t r, int thickness) {
  if (!bgr || width <= 0 || height <= 0 || box_w <= 0 || box_h <= 0) {
    return -1;
  }

  const int x1 = std::max(0, x);
  const int y1 = std::max(0, y);
  const int x2 = std::min(width - 1, x + box_w);
  const int y2 = std::min(height - 1, y + box_h);
  const int t = std::max(1, thickness);
  const int stride = width * 3;

  auto set_pixel = [&](int px, int py) {
    if (px >= 0 && px < width && py >= 0 && py < height) {
      int idx = py * stride + px * 3;
      bgr[idx] = b;
      bgr[idx + 1] = g;
      bgr[idx + 2] = r;
    }
  };

  for (int px = x1; px <= x2; ++px) {
    for (int dt = 0; dt < t; ++dt) {
      set_pixel(px, y1 + dt);
      set_pixel(px, y2 - dt);
    }
  }

  for (int py = y1; py <= y2; ++py) {
    for (int dt = 0; dt < t; ++dt) {
      set_pixel(x1 + dt, py);
      set_pixel(x2 - dt, py);
    }
  }

  const int cx = x1 + (x2 - x1) / 2;
  const int cy = y1 + (y2 - y1) / 2;
  const int arm = 6;
  for (int d = -arm; d <= arm; ++d) {
    set_pixel(cx + d, cy);
    set_pixel(cx, cy + d);
  }

  return 0;
}

} // extern "C"
