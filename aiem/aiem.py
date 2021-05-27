from functools import partial
from types import SimpleNamespace
from numpy import integer, floating, complexfloating
from mpmath import (
    mpmathify, almosteq, mpf, mpc, sin, cos, fabs, sqrt, gamma, besselk, ln, exp
)


class EMWave(object):
    _c0 = mpmathify(299792458)  # Speed of light (m/s) in vacuum
    _h = mpmathify(6.62607015e-34)  # Planck's constant (J.s)

    def __init__(
            self,
            wavelength  # {metre}
    ):
        if isinstance(wavelength, (int, float, integer, floating, mpf)):
            self._wavelength = mpmathify(wavelength)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'wavelength'".format(
                    wavelength
                )
            )

    @classmethod
    def get_c0(cls):
        return cls._c0

    @classmethod
    def get_h(cls):
        return cls._h

    @classmethod
    def from_wavelength(
            cls,
            wavelength  # {metre}
    ):
        return cls(wavelength=wavelength)

    @classmethod
    def from_frequency(
            cls,
            frequency  # {Hz}
    ):
        if isinstance(frequency, (int, float, integer, floating, mpf)):
            wl = cls.get_c0() / mpmathify(frequency)
            return cls(wavelength=wl)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'frequency'".format(frequency)
            )

    @classmethod
    def from_wave_number(
            cls,
            wave_number  # {metre}^{-1}
    ):
        if isinstance(wave_number, (int, float, integer, floating, mpf)):
            wl = mpmathify(1.0) / mpmathify(wave_number)
            return cls(wavelength=wl)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'wave_number'".format(
                    wave_number
                )
            )

    def get_wavelength(self):
        return self._wavelength

    def get_frequency(self):
        return EMWave.get_c0() / self.get_wavelength()

    def get_wave_number(self):
        return mpmathify(1.0) / self.get_wavelength()

    def get_momentum(self):
        return EMWave.get_h() / self.get_wavelength()

    def get_energy(self):
        return (EMWave.get_h() * EMWave.get_c0()) / self.get_wavelength()


class AdvancedIntegralEquationModel(object):
    _phi_i = mpmathify(0)
    _mu_r = mpmathify(1)

    def __init__(
            self,
            theta_i,  # type: float # (radian)
            theta_s,  # type: float # (radian)
            phi_s,  # type: float # (radian)
            wave,  # type: EMWave # k
            nl_cor,  # type: float # kl / k
            nh_rms,  # type: float # ks / k
            eps_r,  # type: complex # (err + eri j)
            surf_type,  # type: float, int # itype
    ):
        if isinstance(theta_i, (int, float, integer, floating, mpf)):
            self._theta_i = mpmathify(theta_i)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'theta_i'".format(
                    theta_i
                )
            )

        if isinstance(theta_s, (int, float, integer, floating, mpf)):
            self._theta_s = mpmathify(theta_s)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'theta_s'".format(
                    theta_s
                )
            )

        if isinstance(phi_s, (int, float, integer, floating, mpf)):
            self._phi_s = mpmathify(phi_s)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'phi_s'".format(
                    phi_s
                )
            )

        if isinstance(wave, EMWave):
            self._wave = wave
        else:
            raise TypeError(
                """
                Invalid type '{}', for parameter 'wave', expected EMWave object.
                """.format(type(wave))
            )

        if isinstance(nl_cor, (int, float, integer, floating, mpf)):
            self._nl_cor = mpmathify(nl_cor)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'nl_cor'".format(
                    nl_cor
                )
            )

        if isinstance(nh_rms, (int, float, integer, floating, mpf)):
            self._nh_rms = mpmathify(nh_rms)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'nh_rms'".format(
                    nh_rms
                )
            )

        if isinstance(eps_r, (complex, complexfloating, mpc)):
            self._eps_r = mpmathify(eps_r)
        else:
            raise ValueError(
                "Invalid value {}, for parameter 'eps_r'".format(
                    eps_r
                )
            )

        self._eps_r = mpmathify(eps_r)

        assert isinstance(
            surf_type, (int, str)
        ), 'Surface Type argument must be a valid <int> or <str>'
        if surf_type == 1 or surf_type.lower() == 'gauss':
            self._surf_type = 1
        elif surf_type == 2 or surf_type.lower() == 'exp':
            self._surf_type = 2
        elif surf_type == 3 or surf_type.lower() == 'tf_exp':
            self._surf_type = 3
        else:
            raise NotImplementedError(
                "Unknown surface type {}!!".format(surf_type)
            )
        self.cache = SimpleNamespace(state=False)

    def _init_cache(self):
        # Initial Caching
        self.cache.k = self.get_wave().get_wave_number()
        self.cache.ks = self.cache.k * self.get_nh_rms()
        self.cache.kl = self.cache.k * self.get_nl_cor()

        self.cache.sin_ti = sin(self.get_theta_i())  # si
        self.cache.cos_ti = cos(self.get_theta_i())  # cs
        self.cache.sin_ts = sin(self.get_theta_s())  # sis
        self.cache.cos_ts = cos(self.get_theta_s())  # css
        self.cache.sin_pi = sin(self.get_phi_i())
        self.cache.cos_pi = cos(self.get_phi_i())
        self.cache.sin_ps = sin(self.get_phi_s())  # sin_ps
        self.cache.cos_ps = cos(self.get_phi_s())  # cos_ps

        self.cache.sti2 = self.cache.sin_ti ** 2  # si2
        self.cache.sts2 = self.cache.sin_ts ** 2  # sis2
        self.cache.cti2 = self.cache.cos_ti ** 2  # cs2
        self.cache.cts2 = self.cache.cos_ts ** 2  # css2

        self.cache.ks2 = self.cache.ks ** 2
        self.cache.kl2 = self.cache.kl ** 2

        iter_factor = self.cache.ks * (
            (self.cache.cos_ti + self.cache.cos_ts) ** 2
        )
        n_iter = 1
        term_prev = mpmathify(0)
        term_curr = iter_factor

        while not (almosteq(term_prev, term_curr)):
            n_iter += 1
            term_prev = term_curr
            term_curr *= iter_factor / n_iter
        
        def calc_spectra(i, c):
            n = mpmathify(i + 1)
            big_k = c.kl * (
                sqrt(
                    (
                        ((c.sin_ts * c.cos_ps) - (c.sin_ti * c.cos_pi)) ** 2
                    ) + (
                        ((c.sin_ts * c.sin_ps) - (c.sin_ti * c.sin_pi)) ** 2
                    )
                )
            )
            surf_type = c.get_surface_type()
            if surf_type == 1:
                term = (c.kl2 / (2 * n)) * (
                    exp(
                        (-1 * (big_k ** 2)) / (4 * n)
                    )
                )
            elif surf_type == 2:
                term = (
                       (c.kl / n) ** 2) * (
                       (1 + ((big_k / n) ** 2)) ** mpmathify(-1.5)
                )
            elif surf_type == 3:
                if almosteq(big_k, 0):
                    term = (c.kl ** 2) / ((3 * n) - 2)
                else:
                    t = mpmathify(1.5) * n
                    g = gamma(t)
                    bk = ln(besselk((1 - t), big_k))
                    co_eff = (c.kl ** 2) * ((big_k / 2) ** (t - 1))
                    term = co_eff * exp((bk - g))
            else:
                raise NotImplementedError(
                    "Unknown surface type {}!!".format(surf_type)
                )
            return term

        spectra = map(partial(calc_spectra, c=self.cache), range(n_iter))
        eps_r = self.get_eps_r()
        mu_r = self.get_mu_r()
        stem = sqrt((eps_r * mu_r) - self.cache.sti2)
        r_vv_i = (
            (eps_r * self.cache.cos_ti) - stem
        ) / (
            (eps_r * self.cache.cos_ti) + stem
        )
        r_hh_i = (
            (mu_r * self.cache.cos_ti) - stem
        ) / (
            (mu_r * self.cache.cos_ti) + stem
        )
        r_vh_i = (r_vv_i - r_hh_i) / 2

        self.cache.n_iter = n_iter
        self.cache.spectra = spectra
        self.cache.r_vv_i = r_vv_i
        self.cache.r_hh_i = r_hh_i
        self.cache.r_vh_i = r_vh_i
        self.cache.state = 1

    def _expal(self, q):
        q = mpmathify(q)
        cache = self.cache
        expalresult = exp(
            -cache.ks2 * (q ** 2.0 - q * (cache.cos_ts - cache.cos_ti))
        )
        return expalresult

    def _spectra_foo(self):
        cache = self.cache
        if not cache.state:
            self._init_cache()
        eps_r = self.get_eps_r()
        mu_r = self.get_mu_r()

        csl = (
            sqrt(
                1 + (
                    cache.cos_ti * cache.cos_ts
                ) - (
                    cache.sin_ti * cache.sin_ts * cache.cos_ps
                )
            )
        ) / (sqrt(2))
        sil = sqrt(1 - (csl ** 2))
        stem_l = sqrt((eps_r * mu_r) - (sil ** 2))
        r_vv_l = ((eps_r * csl) - stem_l) / ((eps_r * csl) + stem_l)
        r_hh_l = ((mu_r * csl) - stem_l) / ((mu_r * csl) + stem_l)
        r_vh_l = (r_vv_l - r_hh_l) / 2

        rv0 = (sqrt(eps_r) - 1) / (sqrt(eps_r) + 1)
        rh0 = -(sqrt(eps_r) - 1) / (sqrt(eps_r) + 1)
        f_tv = 8 * (rv0 ^ 2) * cache.sti2 * (
            cache.cos_ti + sqrt(eps_r - cache.sti2)
        ) / (
            cache.cos_ti * (sqrt(eps_r - cache.sti2))
        )
        f_th = -8 * (rh0 ^ 2) * cache.sti2 * (
            cache.cos_ti + sqrt(eps_r - cache.sti2)
        ) / (
            cache.cos_ti * (sqrt(eps_r - cache.sti2))
        )
        st0v = 1 / (
            (fabs(1 + 8 * rv0 / (cache.cos_ti * f_tv))) ** 2
        )
        st0h = 1 / (
            (fabs(1 + 8 * rv0 / (cache.cos_ti * f_th))) ** 2
        )

        sum_a = mpmathify(0)
        sum_b = mpmathify(0)
        sum_c = mpmathify(0)
        i_factor = mpmathify(1)

        for j in range(cache.n_iter):
            n = mpmathify(j + 1)
            i_factor *= 1 / mpmathify(n)

            sum_a += i_factor * (
                (cache.ks * cache.cos_ti) ** (2 * n)
            ) * cache.spectra[j]

            sum_b += i_factor * (
                    (cache.ks * cache.cos_ti) ** (2 * n)
            ) * (
                fabs(
                    f_tv + (2 ** (n + 2)) * rv0 / cache.cos_ti / exp(
                        (cache.ks * cache.cos_ti) ** 2
                    )
                ) ** 2
            ) * cache.spectra[j]

            sum_c += i_factor * ((cache.ks * cache.cos_ti) ** (2 * n)) * (
                fabs(
                    f_tv + (2 ** (n + 2)) * rv0 / cache.cos_ti * exp(
                        (cache.ks * cache.cos_ti) ** 2
                    )
                ) ** 2
            ) * cache.spectra[j]

        stv = (fabs(f_tv) ** 2) * sum_a / sum_b
        sth = (fabs(f_th) ** 2) * sum_a / sum_c
        tfv = 1 - (stv / st0v)
        tfh = 1 - (sth / st0h)

        if tfv < mpmathify(0):
            tfv = mpmathify(0)

        if tfh < mpmathify(0):
            tfh = mpmathify(0)

        rvv = cache.r_vv_i + (r_vv_l - cache.r_vv_i) * tfv
        rhh = cache.r_hh_i + (r_hh_l - cache.r_hh_i) * tfh
        rvh = (rvv - rhh) / mpmathify(2.0)

        zxx = -(
            cache.sin_ts * cache.cos_ps - cache.sin_ti
        ) / (cache.cos_ts + cache.cos_ti)
        zyy = -(
            cache.sin_ts * cache.sin_ps
        ) / (cache.cos_ts + cache.cos_ti)
        d2 = sqrt(
            (zxx * cache.cos_ti - cache.sin_ti) ** 2 + zyy ** 2
        )
        h_sn_v = -(cache.cos_ti * cache.cos_ps + cache.sin_ti * (
            zxx * cache.cos_ps + zyy * cache.sin_ps)
        )
        v_sn_h = cache.cos_ts * cache.cos_ps - zxx * cache.sin_ts
        h_sn_h = -cache.sin_ps
        v_sn_v = zyy * cache.cos_ti * cache.sin_ts + cache.cos_ts * (
            zyy * cache.cos_ps * cache.sin_ti - (
                cache.cos_ti + zxx * cache.sin_ti
            ) * cache.sin_ps
        )

        h_sn_t = (
            -(cache.cti2 + cache.sti2) * cache.sin_ps * (
                -cache.sin_ti + cache.cos_ti * zxx
            ) + cache.cos_ps * (
                cache.cos_ti + cache.sin_ti * zxx
            ) * zyy + cache.sin_ti * cache.sin_ps * (zyy ** 2)
        ) / d2

        h_sn_d = (
            -(cache.cos_ti + cache.sin_ti * zxx) * (
                (-cache.cos_ps * cache.sin_ti) + (
                     (
                         cache.cos_ti * cache.cos_ps * zxx
                     ) + (
                         cache.cos_ti * cache.sin_ps * zyy
                     )
                )
            )
        ) / d2

        v_sn_t = (
            (
                cache.cti2 + cache.sti2
            ) * (
                -cache.sin_ti + cache.cos_ti * zxx
            ) * (
                cache.cos_ps * cache.cos_ts - cache.sin_ts * zxx
            ) + cache.cos_ts * cache.sin_ps * (
                cache.cos_ti + cache.sin_ti * zxx
            ) * zyy - (
                (
                    cache.cos_ps * cache.cos_ts * cache.sin_ti
                ) + (
                    cache.cos_ti * cache.sin_ts
                )
            ) * (
                zyy ** 2
            )
        ) / d2

        v_sn_d = -(cache.cos_ti + cache.sin_ti * zxx) * (
            cache.sin_ti * cache.sin_ts * zyy - cache.cos_ts * (
                (
                    cache.sin_ti * cache.sin_ps
                ) - (
                    cache.cos_ti * cache.sin_ps * zxx
                ) + (
                    cache.cos_ti * cache.cos_ps * zyy
                )
            )
        ) / d2

        fhh = (1 - rhh) * h_sn_v + (1 + rhh) * v_sn_h - (
            h_sn_t + v_sn_d
        ) * (rhh + rvv) * (zyy / d2)

        fvv = -(
            (1 - rvv) * h_sn_v + (1 + rvv) * v_sn_h
        ) + (h_sn_t + v_sn_d) * (
            rhh + rvv
        ) * (zyy / d2)

        fhv = -(1 + rvv) * h_sn_h + (1 - rvv) * v_sn_v + (
            h_sn_d - v_sn_t
        ) * (rhh + rvv) * (zyy / d2)

        fvh = -(1 + rhh) * h_sn_h + (1 - rhh) * v_sn_v + (
                h_sn_d - v_sn_t
        ) * (rhh + rvv) * (zyy / d2)

        n_sum = 0
        k_factor = 1
        for i in range(cache.n_iter):
            n = i + 1
            k_factor *= (
                cache.ks2 * (cache.cos_ti + cache.cos_ts) ** 2
            ) / n
            n_sum += k_factor * cache.spectra[i]

        exp_k = exp(
            -cache.ks2 * (cache.cos_ts + cache.cos_ti) ** 2
        ) * n_sum
        k_term = [
            (0.5 * exp_k * fabs(fvv) ** 2),
            (0.5 * exp_k * fabs(fhh) ** 2),
            (0.5 * exp_k * fabs(fhv) ** 2),
            (0.5 * exp_k * fabs(fvh) ** 2)
        ]
        qq1 = cache.cos_ti
        qq2 = cache.cos_ts
        qq3 = sqrt(eps_r - cache.sti2)
        qq4 = sqrt(eps_r - cache.sts2)

        fvaupi = self.favv(
            -cache.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fvadni = self.favv(
            -cache.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fvaups = self.favv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fvadns = self.favv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fvbupi = self.fbvv(
            -cache.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fvbdni = self.fbvv(
            -cache.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fvbups = self.fbvv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fvbdns = self.fbvv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)
        fhaupi = self.fahh(
            -cache.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fhadni = self.fahh(
            -cache.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fhaups = self.fahh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fhadns = self.fahh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fhbupi = self.fbhh(
            -cache.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fhbdni = self.fbhh(
            -cache.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fhbups = self.fbhh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fhbdns = self.fbhh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)
        fhvaupi = self.fahv(
            -cache.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fhvadni = self.fahv(
            -cache.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fhvaups = self.fahv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fhvadns = self.fahv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fhvbupi = self.fbhv(
            -cache.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fhvbdni = self.fbhv(
            -cache.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fhvbups = self.fbhv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fhvbdns = self.fbhv(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq4, -
            qq4,
            qq4
        ) * self._expal(-qq4)
        fvhaupi = self.favh(
            -cache.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fvhadni = self.favh(
            -cache.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fvhaups = self.favh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fvhadns = self.favh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fvhbupi = self.fbvh(
            -cache.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fvhbdni = self.fbvh(
            -cache.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fvhbups = self.fbvh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fvhbdns = self.fbvh(
            -cache.sin_ts * cache.cos_ps,
            -cache.sin_ts * cache.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)

        for j in range(cache.n_iter):
            idx = j + 1
            ivv = ((cache.cos_ti + cache.cos_ts) ** idx) * fvv * exp(
                -cache.ks2*cache.cos_ti*cache.cos_ts
            ) + (
                0.25 * (
                    fvaupi*(
                        (cache.cos_ts - qq1) ** idx
                    ) + fvadni * (
                        (cache.cos_ts + qq1) ** idx
                    ) + fvaups * (
                        (cache.cos_ti + qq2) ** idx
                    ) + fvadns * (
                        (cache.cos_ti - qq2) ** idx
                    ) + fvbupi * (
                        (cache.cos_ts - qq3) ** idx
                    ) + fvbdni * (
                        (cache.cos_ts + qq3) ** idx
                    ) + fvbups * (
                        (cache.cos_ti + qq4) ** idx
                    ) + fvbdns * (
                        (cache.cos_ti - qq4) ** idx
                    )
                )
            )

            ihh = ((cache.cos_ti + cache.cos_ts) ** idx) * fhh * exp(
                -cache.ks2*cache.cos_ti*cache.cos_ts
            ) + (
                0.25 * (
                    fhaupi * (
                        (cache.cos_ts - qq1) ** idx
                    ) + fhadni * (
                        (cache.cos_ts + qq1) ** idx
                    ) + fhaups * (
                        (cache.cos_ti + qq2) ** idx
                    ) + fhadns * (
                        (cache.cos_ti - qq2) ** idx
                    ) + fhbupi * (
                        (cache.cos_ts - qq3) ** idx
                    ) + fhbdni * (
                        (cache.cos_ts + qq3) ** idx
                    ) + fhbups * (
                        (cache.cos_ti + qq4) ** idx
                    ) + fhbdns * (
                        (cache.cos_ti - qq4) ** idx
                    )
                )
            )

            ihv = ((cache.cos_ti + cache.cos_ts) ** idx) * fhv * exp(
                -cache.ks2*cache.cos_ti*cache.cos_ts
            ) + (
                0.25 * (
                    fhvaupi*(
                        (cache.cos_ts - qq1) ** idx
                    ) + fhvadni * (
                        (cache.cos_ts + qq1) ** idx
                    ) + fhvaups * (
                        (cache.cos_ti + qq2) ** idx
                    ) + fhvadns * (
                        (cache.cos_ti - qq2) ** idx
                    ) + fhvbupi * (
                        (cache.cos_ts - qq3) ** idx
                    ) + fhvbdni * (
                        (cache.cos_ts + qq3) ** idx
                    ) + fhvbups * (
                        (cache.cos_ti + qq4) ** idx
                    ) + fhvbdns * (
                        (cache.cos_ti-qq4) ** idx
                    )
                )
            )

            ivh = ((cache.cos_ti + cache.cos_ts) ** idx) * fvh * exp(
                -cache.ks2*cache.cos_ti*cache.cos_ts
            ) + (
                0.25 * (
                    fvhaupi*(
                        (cache.cos_ts - qq1) ** idx
                    ) + fvhadni * (
                        (cache.cos_ts + qq1) ** idx
                    ) + fvhaups * (
                        (cache.cos_ti + qq2) ** idx
                    ) + fvhadns * (
                        (cache.cos_ti - qq2) ** idx
                    ) + fvhbupi * (
                        (cache.cos_ts - qq3) ** idx
                    ) + fvhbdni * (
                        (cache.cos_ts + qq3) ** idx
                    ) + fvhbups * (
                        (cache.cos_ti + qq4) ** idx
                    ) + fvhbdns * (
                        (cache.cos_ti - qq4) ** idx
                    )
                )
            )

    def set_phi_i(self, phi_i):
        if isinstance(phi_i, (int, float, integer, floating, mpf)):
            self._phi_i = mpmathify(phi_i)
            self._init_cache()
        else:
            raise ValueError(
                "Invalid value {}, of 'phi_i'".format(
                    phi_i
                )
            )

    def set_mu_r(self, mu_r):
        if isinstance(mu_r, (int, float, integer, floating, mpf)):
            self._mu_r = mpmathify(mu_r)
            self._init_cache()
        else:
            raise ValueError(
                "Invalid value {}, of 'mu_r'".format(
                    mu_r
                )
            )

    def get_phi_i(self):
        return self._phi_i

    def get_mu_r(self):
        return self._mu_r

    def get_theta_i(self):
        return self._theta_i

    def get_theta_s(self):
        return self._theta_s

    def get_phi_s(self):
        return self._phi_s

    def get_wave(self):
        return self._wave

    def get_nh_rms(self):
        return self._nh_rms

    def get_nl_cor(self):
        return self._nl_cor

    def get_surface_type(self):
        return self._surf_type

    def get_eps_r(self):
        return self._eps_r

    def fahh(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache

        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v

        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)

        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)

        c1 = -cache.cos_ps * (-1.0 - zx * zxp) + cache.sin_ps * zxp * zy
        c2 = -cache.cos_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ps * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v * 
            zy * zyp
        )
        c3 = -cache.cos_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) + cache.sin_ps * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        )
        c4 = -cache.cos_ts * cache.sin_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * zx - cache.sin_ti * zx * zxp - cache.sin_ti * zy * 
            zyp
        )
        c5 = -cache.cos_ts * cache.sin_ps * (
            -v * zx + v * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            q + u * zxp + v * zy
        ) + cache.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -cache.cos_ts * cache.sin_ps * (
            -u * zyp + q * zx * zyp) - cache.cos_ps * cache.cos_ts * (
            v * zyp - q * zy * zyp
        ) + cache.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )

        rph = 1.0 + cache.r_hh_i
        rmh = 1.0 - cache.r_hh_i
        ah = rph / qfix
        bh = rmh / qfix
        fahhresult = -bh * (
            -rph * c1 + rmh * c2 + rph * c3
        ) - ah * (
            rmh * c4 + rph * c5 + rmh * c6
        )
        return fahhresult

    def fahv(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)

        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)
        
        b1 = -cache.cos_ts * cache.sin_ps * (
            -1.0 - zx * zxp
        ) - cache.sin_ts * zy - cache.cos_ps * cache.cos_ts * zxp * zy
        b2 = -cache.cos_ts * cache.sin_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ts * (
             -cache.cos_ti * q * zy - q * cache.sin_ti * zxp * zy + q *
             cache.sin_ti * zx * zyp - cache.cos_ti * u * zx * zyp -
             cache.cos_ti * v * zy * zyp
        ) - cache.cos_ps * cache.cos_ts * (
             cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
             cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
             zy * zyp
        )
        b3 = -cache.cos_ts * cache.sin_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        ) + cache.sin_ts * (
             -cache.sin_ti * v * zx + cache.cos_ti * v * zx * zxp +
             cache.sin_ti * u * zy - cache.cos_ti * u * zxp * zy
        )
        b4 = -cache.cos_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) + cache.sin_ps * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        )
        b5 = -cache.cos_ps * (-v * zx + v * zxp) + cache.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -cache.cos_ps * (-u * zyp + q * zx * zyp) + cache.sin_ps * (
            v * zyp - q * zy * zyp
        )
        
        rp = 1.0 + cache.r_vh_i
        rm = 1.0 - cache.r_vh_i
        a = rp / qfix
        b = rm / qfix
        fahvresult = b * (rp * b1 - rm * b2 - rp * b3) + a * (
            rm * b4 + rp * b5 + rm * b6
        )
        return fahvresult

    def favh(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)
        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)

        b1 = -cache.cos_ts * cache.sin_ps * (
            -1.0 - zx * zxp
        ) - cache.sin_ts * zy - cache.cos_ps * cache.cos_ts * zxp * zy
        b2 = -cache.cos_ts * cache.sin_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ts * (
             -cache.cos_ti * q * zy - q * cache.sin_ti * zxp * zy + q *
             cache.sin_ti * zx * zyp - cache.cos_ti * u * zx * zyp -
             cache.cos_ti * v * zy * zyp) - cache.cos_ps * cache.cos_ts * (
                 cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
                 cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti *
                 v * zy * zyp
        )
        b3 = -cache.cos_ts * cache.sin_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        ) + cache.sin_ts * (
            -cache.sin_ti * v * zx + cache.cos_ti * v * zx * zxp +
            cache.sin_ti * u * zy - cache.cos_ti * u * zxp * zy
        )
        b4 = -cache.cos_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) + cache.sin_ps * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        )
        b5 = -cache.cos_ps * (-v * zx + v * zxp) + cache.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -cache.cos_ps * (
            -u * zyp + q * zx * zyp
        ) + cache.sin_ps * (
            v * zyp - q * zy * zyp
        )

        rp = 1.0 + cache.r_vh_i
        rm = 1.0 - cache.r_vh_i
        a = rp / qfix
        b = rm / qfix
        favhresult = b * (
            rp * b4 + rm * b5 + rp * b6
        ) - a * (
            -rm * b1 + rp * b2 + rm * b3
        )
        return favhresult

    def favv(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        kyv = v
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)
        
        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = kyv / (cache.cos_ti + qslp)
        
        c1 = -cache.cos_ps * (-1.0 - zx * zxp) + cache.sin_ps * zxp * zy
        c2 = -cache.cps_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ps * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
            zy * zyp
        )
        c3 = -cache.cos_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) + cache.sin_ps * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        )
        c4 = -cache.cos_ts * cache.sin_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * zx - cache.sin_ti * zx * zxp - cache.sin_ti * zy *
            zyp
        )
        c5 = -cache.cos_ts * cache.sin_ps * (
            -v * zx + v * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            q + u * zxp + v * zy
        ) + cache.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -cache.cos_ts * cache.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            v * zyp - q * zy * zyp
        ) + cache.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )
        rpv = 1.0 + cache.r_vv_i
        rmv = 1.0 - cache.r_vv_i
        av = rpv / qfix
        bv = rmv / qfix
        favvresult = bv * (
            -rpv * c1 + rmv * c2 + rpv * c3
        ) + av * (
            rmv * c4 + rpv * c5 + rmv * c6
        )
        return favvresult

    def fbhh(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)
        
        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)
        
        c1 = -cache.cos_ps * (-1.0 - zx * zxp) + cache.sin_ps * zxp * zy
        c2 = -cache.cos_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ps * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
            zy * zyp
        )
        c3 = -cache.cos_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp) + cache.sin_ps * (
                -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti *
                zy - cache.cos_ti * q * zxp * zy
        )
        c4 = -cache.cos_ts * cache.sin_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * zx - cache.sin_ti * zx * zxp - cache.sin_ti * zy *
            zyp
        )
        c5 = -cache.cos_ts * cache.sin_ps * (
            -v * zx + v * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            q + u * zxp + v * zy
        ) + cache.sin_ts * (q * zx + u * zx * zxp + v * zxp * zy)
        c6 = -cache.cos_ts * cache.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            v * zyp - q * zy * zyp
        ) + cache.sin_ts * (v * zx * zyp - u * zy * zyp)
        rph = 1.0 + cache.r_hh_i
        rmh = 1.0 - cache.r_hh_i
        ah = rph / qfix
        bh = rmh / qfix
        fbhhresult = ah * (
            -rph * c1 * self.get_eps_r() + rmh * c2 + rph * c3
        ) + bh * (
            rmh * c4 + rph * c5 + rmh * c6 / self.get_eps_r()
        )
        return fbhhresult

    def fbhv(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)
        
        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)
        
        b1 = -cache.cos_ts * cache.sin_ps * (
            -1.0 - zx * zxp
        ) - cache.sin_ts * zy - cache.cos_ps * cache.cos_ts * zxp * zy
        b2 = -cache.cos_ts * cache.sin_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * q * zy - q * cache.sin_ti * zxp * zy + q *
            cache.sin_ti * zx * zyp - cache.cos_ti * u * zx * zyp -
            cache.cos_ti * v * zy * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
            zy * zyp
        )
        b3 = -cache.cos_ts * cache.sin_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy) + cache.sin_ts * (
            -cache.sin_ti * v * zx + cache.cos_ti * v * zx * zxp +
            cache.sin_ti * u * zy - cache.cos_ti * u * zxp * zy
        )
        b4 = -cache.cos_ps * (-cache.sin_ti * zyp + cache.cos_ti * zx * zyp) + \
            cache.sin_ps * (
                -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
             )
        b5 = -cache.cos_ps * (-v * zx + v * zxp) + cache.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -cache.cos_ps * (-u * zyp + q * zx * zyp) + cache.sin_ps * (
            v * zyp - q * zy * zyp
        )
        rp = 1.0 + cache.r_vh_i
        rm = 1.0 - cache.r_vh_i
        a = rp / qfix
        b = rm / qfix
        fbhvresult = a * (
            -rp * b1 + rm * b2 + rp * b3 / self.get_eps_r()
        ) - b * (
            rm * b4 * self.get_eps_r() + rp * b5 + rm * b6
        )
        return fbhvresult

    def fbvh(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        v = v
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)

        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)

        b1 = -cache.cos_ts * cache.sin_ps * (
            -1.0 - zx * zxp
        ) - cache.sin_ts * zy - cache.cos_ps * cache.cos_ts * zxp * zy
        b2 = -cache.cos_ts * cache.sin_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * q * zy - q * cache.sin_ti * zxp * zy + q *
            cache.sin_ti * zx * zyp - cache.cos_ti * u * zx * zyp -
            cache.cos_ti * v * zy * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
            zy * zyp
        )
        b3 = -cache.cos_ts * cache.sin_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        ) + cache.sin_ts * (
            -cache.sin_ti * v * zx + cache.cos_ti * v * zx * zxp +
            cache.sin_ti * u * zy - cache.cos_ti * u * zxp * zy
        )
        b4 = -cache.cos_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) + cache.sin_ps * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        )
        b5 = -cache.cos_ps * (
            -v * zx + v * zxp
        ) + cache.sin_ps * (q + u * zxp + v * zy)
        b6 = -cache.cos_ps * (
            -u * zyp + q * zx * zyp
        ) + cache.sin_ps * (v * zyp - q * zy * zyp)
        rp = 1.0 + cache.r_vh_i
        rm = 1.0 - cache.r_vh_i
        a = rp / qfix
        b = rm / qfix
        fbvhresult = -a * (
            rp * b4 + rm * b5 + rp * b6 / self.get_eps_r()
        ) + b * (
            -rm * b1 * self.get_eps_r() + rp * b2 + rm * b3
        )
        return fbvhresult

    def fbvv(self, u, v, q, qslp, qfix, precision=mpmathify(1e-10)):
        u = mpmathify(u)
        v = mpmathify(v)
        q = mpmathify(q)
        qslp = mpmathify(qslp)
        qfix = mpmathify(qfix)
        if self.cache.state < 2:
            if self.cache.state < 1:
                self._init_cache()
            self._spectra_foo()
        cache = self.cache
        kxu = cache.sin_ti + u
        ksxu = cache.sin_ts * cache.cos_ps + u
        ksyv = cache.sin_ts * cache.sin_ps + v
        if fabs((cache.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (cache.cos_ts - qslp)
            zy = (-ksyv) / (cache.cos_ts - qslp)

        if fabs((cache.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (cache.cos_ti + qslp)
            zyp = v / (cache.cos_ti + qslp)

        c1 = -cache.cos_ps * (-1.0 - zx * zxp) + cache.sin_ps * zxp * zy
        c2 = -cache.cos_ps * (
            -cache.cos_ti * q - cache.cos_ti * u * zx - q * cache.sin_ti * zxp -
            cache.sin_ti * u * zx * zxp - cache.cos_ti * v * zyp -
            cache.sin_ti * v * zx * zyp
        ) + cache.sin_ps * (
            cache.cos_ti * u * zy + cache.sin_ti * u * zxp * zy + q *
            cache.sin_ti * zyp - cache.cos_ti * u * zyp + cache.sin_ti * v *
            zy * zyp
        )
        c3 = -cache.cos_ps * (
            cache.sin_ti * u - q * cache.sin_ti * zx - cache.cos_ti * u * zxp +
            cache.cos_ti * q * zx * zxp
        ) + cache.sin_ps * (
            -cache.sin_ti * v + cache.cos_ti * v * zxp + q * cache.sin_ti * zy -
            cache.cos_ti * q * zxp * zy
        )
        c4 = -cache.cos_ts * cache.sin_ps * (
            -cache.sin_ti * zyp + cache.cos_ti * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            -cache.cos_ti - cache.sin_ti * zxp - cache.cos_ti * zy * zyp
        ) + cache.sin_ts * (
            -cache.cos_ti * zx - cache.sin_ti * zx * zxp - cache.sin_ti * zy *
            zyp
        )
        c5 = -cache.cos_ts * cache.sin_ps * (
            -v * zx + v * zxp
        ) - cache.cos_ps * cache.cos_ts * (
            q + u * zxp + v * zy
        ) + cache.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -cache.cos_ts * cache.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - cache.cos_ps * cache.cos_ts * (
            v * zyp - q * zy * zyp
        ) + cache.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )
    
        rpv = 1.0 + cache.r_vv_i
        rmv = 1.0 - cache.r_vv_i
        av = rpv / qfix
        bv = rmv / qfix
        fbvvresult = av * (
            rpv * c1 - rmv * c2 - rpv * c3 / self.get_eps_r()
        ) - bv * (
            rmv * c4 * self.get_eps_r() + rpv * c5 + rmv * c6
        )
        return fbvvresult
