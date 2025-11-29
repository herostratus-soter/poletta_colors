"""
---------------------------------------
MODIFICADO PARA FUNCIONAR SIN COLORAIDE
---------------------------------------

Okhsl class.

Translation to/from Oklab is licenced under MIT by the original author, all
other code also licensed under MIT: Copyright (c) 2021 Isaac Muse.

---- Oklab license ----

Copyright (c) 2021 Björn Ottosson

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

#from coloraide.spaces import Space, RE_DEFAULT_MATCH, Angle, Percent, GamutBound, Cylindrical
#from coloraide.spaces.srgb.base import lin_srgb, gam_srgb
#from coloraide.spaces.oklab import Oklab
#from coloraide import util
#from coloraide import Color as ColorOrig

#coloraide no se va a usar porque se usaban funciones de versiones antiguas


import re
import math
import sys
import copy

FLT_MAX = sys.float_info.max

K_1 = 0.206
K_2 = 0.03
K_3 = (1.0 + K_1) / (1.0 + K_2)

# funciones complementarias por falta de coloraide------------------------

def nth_root(x, n):
    """Raíz n con signo conservado."""
    return math.copysign(abs(x) ** (1.0 / n), x)

def constrain_hue(h):
    """Asegura que el ángulo de tono esté entre 0 y 360."""
    return h % 360

def no_nan(x):
    """Reemplaza NaN por 0."""
    return 0 if math.isnan(x) else x
  
#- - - - - - - - - - - - - - - - - - 
  
def lin_srgb(rgb):
    """Convierte sRGB (gamma) a lineal."""
    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return [f(c) for c in rgb]

def gam_srgb(rgb):
    """Convierte lineal a sRGB (gamma corregido)."""
    def f(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055
    return [f(c) for c in rgb]

#fin funciones complementarias -------------------------------------------------------------
  
  
  
  

def toe(x):
    """Toe function for L_r."""

    return 0.5 * (K_3 * x - K_1 + math.sqrt((K_3 * x - K_1) * (K_3 * x - K_1) + 4 * K_2 * K_3 * x))


def toe_inv(x):
    """Inverse toe function for L_r."""

    return (x * x + K_1 * x) / (K_3 * (x + K_2))


def to_st(cusp):
    """To ST."""

    l, c = cusp
    return c / l, c / (1 - l)


def get_st_mid(a, b):
    """
    Returns a smooth approximation of the location of the cusp.

    This polynomial was created by an optimization process.
    It has been designed so that S_mid < S_max and T_mid < T_max.
    """

    s = 0.11516993 + 1.0 / (
        7.44778970 + 4.15901240 * b +
        a * (
            -2.19557347 + 1.75198401 * b +
            a * (
                -2.13704948 - 10.02301043 * b +
                a * (
                    -4.24894561 + 5.38770819 * b + 4.69891013 * a
                )
            )
        )
    )

    t = 0.11239642 + 1.0 / (
        1.61320320 - 0.68124379 * b +
        a * (
            0.40370612 + 0.90148123 * b +
            a * (
                -0.27087943 + 0.61223990 * b +
                a * (
                    0.00299215 - 0.45399568 * b - 0.14661872 * a
                )
            )
        )
    )

    return s, t


def find_cusp(a, b):
    """
    Finds L_cusp and C_cusp for a given hue.

    `a` and `b` must be normalized so `a^2 + b^2 == 1`.
    """

    # First, find the maximum saturation (saturation `S = C/L`)
    s_cusp = compute_max_saturation(a, b)

    # Convert to linear sRGB to find the first point where at least one of r, g or b >= 1:
    r, g, b = oklab_to_linear_srgb([1, s_cusp * a, s_cusp * b])
    l_cusp = nth_root(1.0 / max(max(r, g), b), 3)
    c_cusp = l_cusp * s_cusp

    return l_cusp , c_cusp


def find_gamut_intersection(a, b, l1, c1, l0, cusp=None):
    """
    Finds intersection of the line.

    Defined by the following:

    ```
    L = L0 * (1 - t) + t * L1
    C = t * C1
    ```

    `a` and `b` must be normalized so `a^2 + b^2 == 1`.
    """

    if cusp is None:
        cusp = get_cs([l1, a, b])

    # Find the intersection for upper and lower half seprately
    if ((l1 - l0) * cusp[1] - (cusp[0] - l0) * c1) <= 0.0:
        #Lower half
        t = cusp[1] * l0 / (c1 * cusp[0] + cusp[1] * (l0 - l1))
    else:
        # Upper half

        # First intersect with triangle
        t = cusp[1] * (l0 - 1.0) / (c1 * (cusp[0] - 1.0) + cusp[1] * (l0 - l1))

        # Then one step Halley's method
        dl = l1 - l0
        dc = c1

        k_l = +0.3963377774 * a + 0.2158037573 * b
        k_m = -0.1055613458 * a - 0.0638541728 * b
        k_s = -0.0894841775 * a - 1.2914855480 * b

        l_dt = dl + dc * k_l
        m_dt = dl + dc * k_m
        s_dt = dl + dc * k_s

        # If higher accuracy is required, 2 or 3 iterations of the following block can be used:
        L = l0 * (1.0 - t) + t * l1
        C = t * c1

        l_ = L + C * k_l
        m_ = L + C * k_m
        s_ = L + C * k_s

        l = l_ ** 3
        m = m_ ** 3
        s = s_ ** 3

        ldt = 3 * l_dt * l_ * l_
        mdt = 3 * m_dt * m_ * m_
        sdt = 3 * s_dt * s_ * s_

        ldt2 = 6 * l_dt * l_dt * l_
        mdt2 = 6 * m_dt * m_dt * m_
        sdt2 = 6 * s_dt * s_dt * s_

        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
        r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
        r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2

        u_r = r1 / (r1 * r1 - 0.5 * r * r2)
        t_r = -r * u_r

        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s - 1
        g1 = -1.2684380046 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
        g2 = -1.2684380046 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2

        u_g = g1 / (g1 * g1 - 0.5 * g * g2)
        t_g = -g * u_g

        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1
        b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt
        b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2

        u_b = b1 / (b1 * b1 - 0.5 * b * b2)
        t_b = -b * u_b

        t_r = t_r if u_r >= 0.0 else FLT_MAX
        t_g = t_g if u_g >= 0.0 else FLT_MAX
        t_b = t_b if u_b >= 0.0 else FLT_MAX

        t += min(t_r, min(t_g, t_b))

    return t


def get_cs(lab):
    l, a, b = lab

    cusp = find_cusp(a, b)

    c_max = find_gamut_intersection(a, b, l, 1, l, cusp)
    st_max = to_st(cusp)

    # Scale factor to compensate for the curved part of gamut shape:
    k = c_max / min((l * st_max[0]), (1 - l) * st_max[1])

    st_mid = get_st_mid(a, b)

    # Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
    c_a = l * st_mid[0]
    c_b = (1.0 - l) * st_mid[1]
    c_mid = 0.9 * k * math.sqrt(math.sqrt(1.0 / (1.0 / (c_a * c_a * c_a * c_a) + 1.0 / (c_b * c_b * c_b * c_b))))

    # For `C_0`, the shape is independent of hue, so `ST` are constant.
    # Values picked to roughly be the average values of `ST`.
    c_a = l * 0.4
    c_b = (1.0 - l) * 0.8

    # Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
    c_0 = math.sqrt(1.0 / (1.0 / (c_a * c_a) + 1.0 / (c_b * c_b)))

    return c_0, c_mid, c_max


def oklab_to_linear_srgb(lab):
    """Convert from Oklab to linear sRGB."""

    l_ = lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2]
    m_ = lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2]
    s_ = lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2]

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    return [
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    ]


def linear_srgb_to_oklab(rgb):

    r, g, b = rgb
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = nth_root(l, 3)
    m_ = nth_root(m, 3)
    s_ = nth_root(s, 3)

    return [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    ]


def compute_max_saturation(a, b):
    """
    Finds the maximum saturation possible for a given hue that fits in sRGB.
    Saturation here is defined as `S = C/L`.
    `a` and `b` must be normalized so `a^2 + b^2 == 1`.
    """

    # Max saturation will be when one of r, g or b goes below zero.

    # Select different coefficients depending on which component goes below zero first.

    if (-1.88170328 * a - 0.80936493 * b) > 1:
        # Red component
        k0 = 1.19086277
        k1 = 1.76576728
        k2 = 0.59662641
        k3 = 0.75515197
        k4 = 0.56771245
        wl = 4.0767416621
        wm = -3.3077115913
        ws = 0.2309699292

    elif (1.81444104 * a - 1.19445276 * b) > 1:
        # Green component
        k0 = 0.73956515
        k1 = -0.45954404
        k2 = 0.08285427
        k3 = 0.12541070
        k4 = 0.14503204
        wl = -1.2684380046
        wm = 2.6097574011
        ws = -0.3413193965

    else:
        # Blue component
        k0 = 1.35733652
        k1 = -0.00915799
        k2 = -1.15130210
        k3 = -0.50559606
        k4 = 0.00692167
        wl = -0.0041960863
        wm = -0.7034186147
        ws = 1.7076147010

    # Approximate max saturation using a polynomial:
    sat = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

    # Do one step Halley's method to get closer.
    # This gives an error less than 10e6, except for some blue hues where the `dS/dh` is close to infinite.
    # This should be sufficient for most applications, otherwise do two/three steps.

    k_l = 0.3963377774 * a + 0.2158037573 * b
    k_m = -0.1055613458 * a - 0.0638541728 * b
    k_s = -0.0894841775 * a - 1.2914855480 * b

    l_ = 1.0 + sat * k_l
    m_ = 1.0 + sat * k_m
    s_ = 1.0 + sat * k_s

    l = l_ * l_ * l_
    m = m_ * m_ * m_
    s = s_ * s_ * s_

    l_ds = 3.0 * k_l * l_ * l_
    m_ds = 3.0 * k_m * m_ * m_
    s_ds = 3.0 * k_s * s_ * s_

    l_ds2 = 6.0 * k_l * k_l * l_
    m_ds2 = 6.0 * k_m * k_m * m_
    s_ds2 = 6.0 * k_s * k_s * s_

    f = wl * l + wm * m + ws * s
    f1 = wl * l_ds + wm * m_ds + ws * s_ds
    f2 = wl * l_ds2 + wm * m_ds2 + ws * s_ds2

    sat = sat - f * f1 / (f1 * f1 - 0.5 * f * f2)

    return sat


def okhsl_to_oklab(hsl):
    """Convert Okhsl to sRGB."""

    h, s, l = hsl
    s /= 100
    l /= 100
    h = no_nan(h) / 360.0

    L = toe_inv(l)
    a = b = 0

    if L != 0 and L != 1 and s != 0:
        a_ = math.cos(2.0 * math.pi * h)
        b_ = math.sin(2.0 * math.pi * h)

        c_0, c_mid, c_max = get_cs([L, a_, b_])

        # Interpolate the three values for C so that:
        # At s=0: dC/ds = C_0, C=0
        # At s=0.8: C=C_mid
        # At s=1.0: C=C_max

        mid = 0.8
        mid_inv = 1.25

        if s < mid:
            t = mid_inv * s
            k_0 = 0
            k_1 = mid * c_0
            k_2 = (1.0 - k_1 / c_mid)

        else:
            t = 5 * (s - 0.8)
            k_0 = c_mid
            k_1 = 0.2 * (c_mid ** 2) * (1.25 ** 2) / c_0
            k_2 = (1.0 - (k_1) / (c_max - c_mid))

        c = k_0 + t * k_1 / (1.0 - k_2 * t)

        a = c * a_
        b = c * b_

    return [L, a, b]


def oklab_to_okhsl(lab):
    """Oklab to Okhsl."""

    c = math.sqrt(lab[1] ** 2 + lab[2] ** 2)

    h = float('nan')
    L = lab[0]
    s = 0

    if c != 0 and L != 0:
        a_ = lab[1] / c
        b_ = lab[2] / c

        h = 0.5 + 0.5 * math.atan2(-lab[2], -lab[1]) / math.pi

        c_0, c_mid, c_max = get_cs([L, a_, b_])

        # Inverse of the interpolation in okhsl_to_srgb:

        mid = 0.8
        mid_inv = 1.25

        if (c < c_mid):
            k_1 = mid * c_0
            k_2 = (1.0 - k_1 / c_mid)

            t = c / (k_1 + k_2 * c)
            s = t * mid

        else:
            k_0 = c_mid
            k_1 = (1.0 - mid) * c_mid * c_mid * mid_inv * mid_inv / c_0
            k_2 = (1.0 - (k_1) / (c_max - c_mid))

            t = (c - k_0) / (k_1 + k_2 * (c - k_0))
            s = mid + (1.0 - mid) * t

    l = toe(L)

    if s == 0:
        h = float('nan')

    return constrain_hue(h * 360), s * 100, l * 100


def okhsv_to_oklab(hsv):
    """Convert from Okhsv to Oklab."""

    h, s, v = hsv
    s /= 100
    v /= 100
    h = no_nan(h) / 360.0

    l = toe_inv(v)
    a = b = 0

    if l != 0 and s != 0:
        a_ = math.cos(2.0 * math.pi * h)
        b_ = math.sin(2.0 * math.pi * h)

        cusp = find_cusp(a_, b_)
        s_max, t_max = to_st(cusp)
        s_0 = 0.5
        k = 1 - s_0 / s_max

        # first we compute L and V as if the gamut is a perfect triangle:

        # L, C when v==1:
        l_v = 1 - s * s_0 / (s_0 + t_max - t_max * k * s)
        c_v = s * t_max * s_0 / (s_0 + t_max - t_max * k * s)

        l = v * l_v
        c = v * c_v

        # then we compensate for both toe and the curved top part of the triangle:
        l_vt = toe_inv(l_v)
        c_vt = c_v * l_vt / l_v

        l_new = toe_inv(l)
        c = c * l_new / l
        l = l_new

        # RGB scale
        rs, gs, bs = oklab_to_linear_srgb([l_vt, a_ * c_vt, b_ * c_vt])
        scale_l = nth_root(1.0 / max(max(rs, gs), max(bs, 0.0)), 3)

        l = l * scale_l
        c = c * scale_l

        a = c * a_
        b = c * b_

    return [l, a, b]


def oklab_to_okhsv(lab):
    """Oklab to Okhsv."""

    c = math.sqrt(lab[1] ** 2 + lab[2] ** 2)
    l = lab[0]

    h = float('nan')
    s = 0
    v = toe(l)

    if c != 0 and l != 0 and l != 1:
        a_ = lab[1] / c
        b_ = lab[2] / c

        h = 0.5 + 0.5 * math.atan2(-lab[2], -lab[1]) / math.pi

        cusp = find_cusp(a_, b_)
        s_max, t_max = to_st(cusp)
        s_0 = 0.5
        k = 1 - s_0 / s_max

        # first we find L_v, C_v, L_vt and C_vt
        t = t_max / (c + l * t_max)
        l_v = t * l
        c_v = t * c

        l_vt = toe_inv(l_v)
        c_vt = c_v * l_vt / l_v

        # we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
        rs, gs, bs = oklab_to_linear_srgb([l_vt, a_ * c_vt, b_ * c_vt])
        scale_l = nth_root(1.0 / max(max(rs, gs), max(bs, 0.0)), 3)

        l = l / scale_l
        c = c / scale_l

        c = c * toe(l) / l
        l = toe(l)

        # we can now compute v and s:
        v = l / l_v
        s = (s_0 + t_max) * c_v / ((t_max * s_0) + t_max * k * c_v)

    if s == 0:
        h = float('nan')

    return [constrain_hue(h * 360), s * 100, v * 100]

def srgb_to_okhsv(srgb):
    """SRGB to Okhsv."""

    return oklab_to_okhsv(linear_srgb_to_oklab(lin_srgb(srgb)))


def okhsv_to_srgb(hsv):
    """Okhsv to sRGB."""

    return gam_srgb(oklab_to_linear_srgb(okhsv_to_oklab(hsv)))


def srgb_to_okhsl(srgb):
    """SRGB to Okhsl."""

    return oklab_to_okhsl(linear_srgb_to_oklab(lin_srgb(srgb)))


def okhsl_to_srgb(hsl):
    """Okhsl to sRGB."""

    return gam_srgb(oklab_to_linear_srgb(okhsl_to_oklab(hsl)))

