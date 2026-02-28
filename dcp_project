import cv2
import numpy as np
import argparse


# =========================
# 1. 暗通道计算
# =========================
def dark_channel(im_float, win=15):
    """
    计算暗通道（Dark Channel）

    参数：
        im_float: float32 图像，范围 [0,1]，shape = (H, W, 3)
                  注意：BGR / RGB 顺序无所谓，因为这里只取 min(R,G,B)
        win: 暗通道窗口大小（通常取 7~21 之间的奇数）

    原理简述：
        暗通道先验认为：在无雾室外自然图像中，除天空区域外，
        局部窗口内总存在某个像素在某个颜色通道上接近 0。
        因此定义暗通道为：
            Dark(x) = min_{y∈Ω(x)} ( min_c I^c(y) )
        即：先对每个像素取 RGB 三通道最小值，再对局部窗口取最小值（相当于形态学腐蚀 erode）。
    """
    # 每个像素取三通道最小值：min_c I^c(x)
    min_rgb = np.min(im_float, axis=2)

    # 对 min_rgb 做窗口最小值滤波：min_{y∈Ω(x)}(...)
    # 形态学腐蚀 (erode) 就是局部最小值滤波
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dark = cv2.erode(min_rgb, kernel)

    return dark


# =========================
# 2. 大气光 A 估计
# =========================
def estimate_atmospheric_light(im_float, dark, top_percent=0.001):
    """
    估计大气光 A（Atmospheric Light）

    参数：
        im_float: float32 图像 [0,1]，shape=(H,W,3)
        dark: 暗通道图，shape=(H,W)
        top_percent: 取暗通道最亮的前 top_percent 比例像素来估计 A
                     常用 0.001（0.1%）或 0.005（0.5%）

    原理简述：
        在有雾图像中，暗通道值较大的区域往往对应“雾很浓/接近大气光”的区域；
        因此先挑出暗通道中最亮的一小部分像素（top p%），
        然后在这些像素中选择原图强度（R+G+B）最大的像素作为 A。
    """
    h, w = dark.shape
    num = max(1, int(h * w * top_percent))  # 至少取 1 个点

    # 拉平成一维，方便取 top 百分位
    dark_vec = dark.reshape(-1)        # shape=(H*W,)
    im_vec = im_float.reshape(-1, 3)   # shape=(H*W,3)

    # argpartition 用于快速找出最大 num 个元素的索引（比全排序快）
    indices = np.argpartition(dark_vec, -num)[-num:]

    # 在候选集合中，找原图亮度最大的像素
    candidates = im_vec[indices]                 # shape=(num,3)
    intensity = np.sum(candidates, axis=1)       # shape=(num,)
    A = candidates[np.argmax(intensity)]         # shape=(3,)

    return A


# =========================
# 3. 估计粗传输图 t(x)
# =========================
def estimate_transmission(im_float, A, win=15, omega=0.95):
    """
    估计粗传输图 t(x)（Transmission）

    参数：
        im_float: float32 图像 [0,1]
        A: 大气光，shape=(3,)
        win: 暗通道窗口大小
        omega: 去雾强度系数（常用 0.8~0.98，越大去雾越强）

    原理简述：
        在 DCP 中，先把图像按 A 归一化：
            I_norm^c(x) = I^c(x) / A^c
        然后对 I_norm 求暗通道：
            dark_norm(x) = min_{y∈Ω(x)} min_c I_norm^c(y)
        则传输图估计为：
            t(x) = 1 - ω * dark_norm(x)
    """
    # 避免除零（虽然 A 一般不会是 0，但做健壮性处理）
    A = np.maximum(A, 1e-6)

    # 广播除法：每个通道除以对应 A^c
    im_norm = im_float / A

    # 归一化后的暗通道
    dark_norm = dark_channel(im_norm, win)

    # 粗传输图
    t = 1.0 - omega * dark_norm

    return t


# =========================
# 4. Guided Filter 细化传输图（替代 soft matting）
# =========================
def guided_filter(I, p, r=40, eps=1e-3):
    """
    Guided Filter（导向滤波）实现，用于细化传输图 t(x)

    参数：
        I: 引导图（通常用原图灰度），float32 [0,1]，shape=(H,W)
        p: 输入需要被滤波的图（这里是粗传输图 t_raw），float32 [0,1]，shape=(H,W)
        r: 滤波半径（窗口尺寸为 (2r+1)*(2r+1)）
        eps: 正则项，防止除零并控制平滑强度（常用 1e-4~1e-2）

    为什么需要细化？
        粗传输图 t_raw 是通过窗口最小值滤波得到的，往往会有“块效应”和边缘不准；
        Guided Filter 能在保持边缘的前提下平滑 t_raw，使去雾结果更自然。

    Guided Filter 核心公式（简述）：
        假设在局部窗口 ω 内，有线性模型：
            q = a * I + b
        通过最小化 (q - p)^2 + eps*a^2 求得 a,b
        然后对 a,b 在窗口内求均值，得到输出 q

    这里用 cv2.blur 做均值滤波（box filter），实现近似 O(N)。
    """
    win = (2 * r + 1, 2 * r + 1)

    # 计算均值
    mean_I = cv2.blur(I, win)
    mean_p = cv2.blur(p, win)
    mean_Ip = cv2.blur(I * p, win)

    # 协方差 cov(I,p) = E(Ip) - E(I)E(p)
    cov_Ip = mean_Ip - mean_I * mean_p

    # 方差 var(I) = E(II) - E(I)^2
    mean_II = cv2.blur(I * I, win)
    var_I = mean_II - mean_I * mean_I

    # 线性系数 a, b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    # 对 a, b 再做均值滤波（窗口内平滑）
    mean_a = cv2.blur(a, win)
    mean_b = cv2.blur(b, win)

    # 输出 q
    q = mean_a * I + mean_b
    return q


# =========================
# 5. 场景辐射恢复 J(x)
# =========================
def recover_scene_radiance(im_float, A, t, t0=0.1):
    """
    根据成像模型恢复无雾图 J(x)

    参数：
        im_float: float32 输入有雾图 [0,1]
        A: 大气光，shape=(3,)
        t: 细化后的传输图，shape=(H,W)
        t0: 传输图下限（防止 t 太小导致噪声放大）

    成像模型：
        I(x) = J(x)*t(x) + A*(1-t(x))
    解得：
        J(x) = (I(x)-A)/t(x) + A

    注意：
        当 t(x) 非常小（雾极浓或天空区域）时，除以 t 会把噪声和误差放大，
        因此把 t 截断到 [t0, 1]。
    """
    t = np.clip(t, t0, 1.0)
    J = (im_float - A) / t[..., None] + A
    return np.clip(J, 0.0, 1.0)


# =========================
# 6. DCP 主流程封装
# =========================
def dehaze_dcp(bgr_uint8, win=15, omega=0.95, t0=0.1, gf_radius=40, gf_eps=1e-3):
    """
    对单张 BGR 图像执行 DCP 去雾

    参数：
        bgr_uint8: 输入 BGR uint8 图像（OpenCV 默认读进来就是 BGR）
        win: 暗通道窗口大小（影响雾估计尺度）
        omega: 去雾强度
        t0: 传输图下限
        gf_radius: Guided Filter 半径（越大越平滑但可能损失局部细节）
        gf_eps: Guided Filter 正则项（越大越平滑）

    返回：
        out_uint8: 去雾后的 BGR uint8 图像
    """
    # uint8 -> float32, 归一化到 [0,1]
    im = bgr_uint8.astype(np.float32) / 255.0

    # (1) 暗通道
    dark = dark_channel(im, win=win)

    # (2) 估计大气光 A
    A = estimate_atmospheric_light(im, dark, top_percent=0.001)

    # (3) 粗传输图
    t_raw = estimate_transmission(im, A, win=win, omega=omega)

    # (4) 细化传输图：用灰度图作为引导
    gray = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_refined = guided_filter(gray, t_raw.astype(np.float32), r=gf_radius, eps=gf_eps)
    t_refined = np.clip(t_refined, 0.0, 1.0)

    # (5) 恢复无雾图
    J = recover_scene_radiance(im, A, t_refined, t0=t0)

    # float -> uint8
    out_uint8 = (J * 255.0).astype(np.uint8)
    return out_uint8


# =========================
# 7. 命令行入口：读图->去雾->保存
# =========================
def main():
    parser = argparse.ArgumentParser(description="暗通道先验(DCP) 单图去雾（带 Guided Filter 细化）")
    parser.add_argument("input", help="输入图片路径，例如 input.jpg")
    parser.add_argument("output", help="输出图片路径，例如 output.jpg")
    parser.add_argument("--win", type=int, default=15, help="暗通道窗口大小(7~21常用)")
    parser.add_argument("--omega", type=float, default=0.95, help="去雾强度系数(0.8~0.98)")
    parser.add_argument("--t0", type=float, default=0.1, help="传输图下限(0.05~0.2)")
    parser.add_argument("--gf_r", type=int, default=40, help="导向滤波半径(20~80)")
    parser.add_argument("--gf_eps", type=float, default=1e-3, help="导向滤波eps(1e-4~1e-2)")
    args = parser.parse_args()

    # 读入 BGR 图像
    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图片：{args.input}")

    # 去雾
    out = dehaze_dcp(
        bgr,
        win=args.win,
        omega=args.omega,
        t0=args.t0,
        gf_radius=args.gf_r,
        gf_eps=args.gf_eps
    )

    # 保存
    ok = cv2.imwrite(args.output, out)
    if not ok:
        raise RuntimeError(f"写入失败：{args.output}")

    print(f"去雾完成，已保存到：{args.output}")


if __name__ == "__main__":
    main()