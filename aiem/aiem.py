from functools import partial
from types import SimpleNamespace
from numpy import integer, floating, complexfloating
from mpmath import (
    mpf, mpc, sin, cos, fabs, sqrt, ln, factorial,
    exp, gamma, besselk, mpmathify, almosteq
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

        spectra = list(map(partial(calc_spectra, c=self.cache), range(n_iter)))
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

        csl = (
            sqrt(
                1 + (
                    self.cache.cos_ti * self.cache.cos_ts
                ) - (
                    self.cache.sin_ti * self.cache.sin_ts * self.cache.cos_ps
                )
            )
        ) / (sqrt(2))
        sil = sqrt(1 - (csl ** 2))
        stem_l = sqrt((eps_r * mu_r) - (sil ** 2))
        r_vv_l = ((eps_r * csl) - stem_l) / ((eps_r * csl) + stem_l)
        r_hh_l = ((mu_r * csl) - stem_l) / ((mu_r * csl) + stem_l)
        # r_vh_l = (r_vv_l - r_hh_l) / 2

        self.cache.n_iter = n_iter
        self.cache.spectra = spectra

        self.cache.r_vv_i = r_vv_i
        self.cache.r_hh_i = r_hh_i
        self.cache.r_vh_i = r_vh_i

        self.cache.r_vv_l = r_vv_l
        self.cache.r_hh_l = r_hh_l
        # self.cache.r_vh_l = r_vh_l

        self.cache.state = True

    def _expal(self, q):
        q = mpmathify(q)
        c = self.cache
        expalresult = exp(
            -c.ks2 * (q ** 2.0 - q * (c.cos_ts - c.cos_ti))
        )
        return expalresult

    def _spectra_foo(self):
        c = self.cache
        if not c.state:
            self._init_cache()
        eps_r = self.get_eps_r()

        rv0 = (sqrt(eps_r) - 1) / (sqrt(eps_r) + 1)
        rh0 = -(sqrt(eps_r) - 1) / (sqrt(eps_r) + 1)
        f_tv = 8 * (rv0 ^ 2) * c.sti2 * (
            c.cos_ti + sqrt(eps_r - c.sti2)
        ) / (
            c.cos_ti * (sqrt(eps_r - c.sti2))
        )
        f_th = -8 * (rh0 ^ 2) * c.sti2 * (
            c.cos_ti + sqrt(eps_r - c.sti2)
        ) / (
            c.cos_ti * (sqrt(eps_r - c.sti2))
        )
        st0v = 1 / (
            (fabs(1 + 8 * rv0 / (c.cos_ti * f_tv))) ** 2
        )
        st0h = 1 / (
            (fabs(1 + 8 * rv0 / (c.cos_ti * f_th))) ** 2
        )

        def calc_abc(k, stash):
            m = mpmathify(k + 1)
            i_factor = 1 / factorial(m)

            a_ = i_factor * (
                    (stash.ks * stash.cos_ti) ** (2 * m)
            ) * stash.spectra[k]

            b_ = i_factor * (
                    (stash.ks * stash.cos_ti) ** (2 * m)
            ) * (
                fabs(
                    f_tv + (2 ** (m + 2)) * rv0 / stash.cos_ti / exp(
                        (stash.ks * stash.cos_ti) ** 2
                    )
                ) ** 2
            ) * stash.spectra[k]

            c_ = i_factor * ((stash.ks * stash.cos_ti) ** (2 * m)) * (
                fabs(
                    f_tv + (2 ** (m + 2)) * rv0 / stash.cos_ti * exp(
                        (stash.ks * stash.cos_ti) ** 2
                    )
                ) ** 2
            ) * stash.spectra[k]
            return [a_, b_, c_]

        abc_list = list(
            map(partial(calc_abc, stash=c), range(c.n_iter))
        )
        sum_a, sum_b, sum_c = list(map(sum, map(list, zip(*abc_list))))

        stv = (fabs(f_tv) ** 2) * sum_a / sum_b
        sth = (fabs(f_th) ** 2) * sum_a / sum_c
        tfv = 1 - (stv / st0v)
        tfh = 1 - (sth / st0h)

        if tfv < mpmathify(0):
            tfv = mpmathify(0)

        if tfh < mpmathify(0):
            tfh = mpmathify(0)

        rvv = c.r_vv_i + (c.r_vv_l - c.r_vv_i) * tfv
        rhh = c.r_hh_i + (c.r_hh_l - c.r_hh_i) * tfh
        # rvh = (rvv - rhh) / mpmathify(2.0)

        zxx = -(
            c.sin_ts * c.cos_ps - c.sin_ti
        ) / (c.cos_ts + c.cos_ti)
        zyy = -(
            c.sin_ts * c.sin_ps
        ) / (c.cos_ts + c.cos_ti)
        d2 = sqrt(
            (zxx * c.cos_ti - c.sin_ti) ** 2 + zyy ** 2
        )
        h_sn_v = -(c.cos_ti * c.cos_ps + c.sin_ti * (
            zxx * c.cos_ps + zyy * c.sin_ps)
        )
        v_sn_h = c.cos_ts * c.cos_ps - zxx * c.sin_ts
        h_sn_h = -c.sin_ps
        v_sn_v = zyy * c.cos_ti * c.sin_ts + c.cos_ts * (
            zyy * c.cos_ps * c.sin_ti - (
                c.cos_ti + zxx * c.sin_ti
            ) * c.sin_ps
        )

        h_sn_t = (
            -(c.cti2 + c.sti2) * c.sin_ps * (
                -c.sin_ti + c.cos_ti * zxx
            ) + c.cos_ps * (
                c.cos_ti + c.sin_ti * zxx
            ) * zyy + c.sin_ti * c.sin_ps * (zyy ** 2)
        ) / d2

        h_sn_d = (
            -(c.cos_ti + c.sin_ti * zxx) * (
                (-c.cos_ps * c.sin_ti) + (
                     (
                         c.cos_ti * c.cos_ps * zxx
                     ) + (
                         c.cos_ti * c.sin_ps * zyy
                     )
                )
            )
        ) / d2

        v_sn_t = (
            (
                c.cti2 + c.sti2
            ) * (
                -c.sin_ti + c.cos_ti * zxx
            ) * (
                c.cos_ps * c.cos_ts - c.sin_ts * zxx
            ) + c.cos_ts * c.sin_ps * (
                c.cos_ti + c.sin_ti * zxx
            ) * zyy - (
                (
                    c.cos_ps * c.cos_ts * c.sin_ti
                ) + (
                    c.cos_ti * c.sin_ts
                )
            ) * (
                zyy ** 2
            )
        ) / d2

        v_sn_d = -(c.cos_ti + c.sin_ti * zxx) * (
            c.sin_ti * c.sin_ts * zyy - c.cos_ts * (
                (
                    c.sin_ti * c.sin_ps
                ) - (
                    c.cos_ti * c.sin_ps * zxx
                ) + (
                    c.cos_ti * c.cos_ps * zyy
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
        
        # def calc_term(t, stash):
        #     n = t + 1
        #     t_factor = (
        #        stash.ks2 * (stash.cos_ti + stash.cos_ts) ** (2 * n)
        #     ) / factorial(n)
        #     return t_factor
        #
        # n_sum = sum(list(map(partial(calc_term, c=c), range(c.n_iter))))
        #
        # exp_k = exp(
        #     -c.ks2 * (c.cos_ts + c.cos_ti) ** 2
        # ) * n_sum
        # k_term = [
        #     (0.5 * exp_k * fabs(fvv) ** 2),
        #     (0.5 * exp_k * fabs(fhh) ** 2),
        #     (0.5 * exp_k * fabs(fhv) ** 2),
        #     (0.5 * exp_k * fabs(fvh) ** 2)
        # ]
        qq1 = c.cos_ti
        qq2 = c.cos_ts
        qq3 = sqrt(eps_r - c.sti2)
        qq4 = sqrt(eps_r - c.sts2)

        fvaupi = self.favv(
            -c.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fvadni = self.favv(
            -c.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fvaups = self.favv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fvadns = self.favv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fvbupi = self.fbvv(
            -c.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fvbdni = self.fbvv(
            -c.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fvbups = self.fbvv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fvbdns = self.fbvv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)
        fhaupi = self.fahh(
            -c.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fhadni = self.fahh(
            -c.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fhaups = self.fahh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fhadns = self.fahh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fhbupi = self.fbhh(
            -c.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fhbdni = self.fbhh(
            -c.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fhbups = self.fbhh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fhbdns = self.fbhh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)
        fhvaupi = self.fahv(
            -c.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fhvadni = self.fahv(
            -c.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fhvaups = self.fahv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fhvadns = self.fahv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fhvbupi = self.fbhv(
            -c.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fhvbdni = self.fbhv(
            -c.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fhvbups = self.fbhv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fhvbdns = self.fbhv(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq4, -
            qq4,
            qq4
        ) * self._expal(-qq4)
        fvhaupi = self.favh(
            -c.sin_ti, 0.0, qq1, qq1, qq1
        ) * self._expal(qq1)
        fvhadni = self.favh(
            -c.sin_ti, 0.0, -qq1, -qq1, qq1
        ) * self._expal(-qq1)
        fvhaups = self.favh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq2,
            qq2,
            qq2
        ) * self._expal(qq2)
        fvhadns = self.favh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq2,
            -qq2,
            qq2
        ) * self._expal(-qq2)
        fvhbupi = self.fbvh(
            -c.sin_ti, 0.0, qq3, qq3, qq3
        ) * self._expal(qq3)
        fvhbdni = self.fbvh(
            -c.sin_ti, 0.0, -qq3, -qq3, qq3
        ) * self._expal(-qq3)
        fvhbups = self.fbvh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            qq4,
            qq4,
            qq4
        ) * self._expal(qq4)
        fvhbdns = self.fbvh(
            -c.sin_ts * c.cos_ps,
            -c.sin_ts * c.sin_ps,
            -qq4,
            -qq4,
            qq4
        ) * self._expal(-qq4)

        for j in range(c.n_iter):
            idx = j + 1
            ivv = ((c.cos_ti + c.cos_ts) ** idx) * fvv * exp(
                -c.ks2*c.cos_ti*c.cos_ts
            ) + (
                0.25 * (
                    fvaupi*(
                        (c.cos_ts - qq1) ** idx
                    ) + fvadni * (
                        (c.cos_ts + qq1) ** idx
                    ) + fvaups * (
                        (c.cos_ti + qq2) ** idx
                    ) + fvadns * (
                        (c.cos_ti - qq2) ** idx
                    ) + fvbupi * (
                        (c.cos_ts - qq3) ** idx
                    ) + fvbdni * (
                        (c.cos_ts + qq3) ** idx
                    ) + fvbups * (
                        (c.cos_ti + qq4) ** idx
                    ) + fvbdns * (
                        (c.cos_ti - qq4) ** idx
                    )
                )
            )

            ihh = ((c.cos_ti + c.cos_ts) ** idx) * fhh * exp(
                -c.ks2*c.cos_ti*c.cos_ts
            ) + (
                0.25 * (
                    fhaupi * (
                        (c.cos_ts - qq1) ** idx
                    ) + fhadni * (
                        (c.cos_ts + qq1) ** idx
                    ) + fhaups * (
                        (c.cos_ti + qq2) ** idx
                    ) + fhadns * (
                        (c.cos_ti - qq2) ** idx
                    ) + fhbupi * (
                        (c.cos_ts - qq3) ** idx
                    ) + fhbdni * (
                        (c.cos_ts + qq3) ** idx
                    ) + fhbups * (
                        (c.cos_ti + qq4) ** idx
                    ) + fhbdns * (
                        (c.cos_ti - qq4) ** idx
                    )
                )
            )

            ihv = ((c.cos_ti + c.cos_ts) ** idx) * fhv * exp(
                -c.ks2*c.cos_ti*c.cos_ts
            ) + (
                0.25 * (
                    fhvaupi*(
                        (c.cos_ts - qq1) ** idx
                    ) + fhvadni * (
                        (c.cos_ts + qq1) ** idx
                    ) + fhvaups * (
                        (c.cos_ti + qq2) ** idx
                    ) + fhvadns * (
                        (c.cos_ti - qq2) ** idx
                    ) + fhvbupi * (
                        (c.cos_ts - qq3) ** idx
                    ) + fhvbdni * (
                        (c.cos_ts + qq3) ** idx
                    ) + fhvbups * (
                        (c.cos_ti + qq4) ** idx
                    ) + fhvbdns * (
                        (c.cos_ti-qq4) ** idx
                    )
                )
            )

            ivh = ((c.cos_ti + c.cos_ts) ** idx) * fvh * exp(
                -c.ks2*c.cos_ti*c.cos_ts
            ) + (
                0.25 * (
                    fvhaupi*(
                        (c.cos_ts - qq1) ** idx
                    ) + fvhadni * (
                        (c.cos_ts + qq1) ** idx
                    ) + fvhaups * (
                        (c.cos_ti + qq2) ** idx
                    ) + fvhadns * (
                        (c.cos_ti - qq2) ** idx
                    ) + fvhbupi * (
                        (c.cos_ts - qq3) ** idx
                    ) + fvhbdni * (
                        (c.cos_ts + qq3) ** idx
                    ) + fvhbups * (
                        (c.cos_ti + qq4) ** idx
                    ) + fvhbdns * (
                        (c.cos_ti - qq4) ** idx
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
        c = self.cache

        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v

        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)

        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)

        c1 = -c.cos_ps * (-1.0 - zx * zxp) + c.sin_ps * zxp * zy
        c2 = -c.cos_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ps * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v * 
            zy * zyp
        )
        c3 = -c.cos_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) + c.sin_ps * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        )
        c4 = -c.cos_ts * c.sin_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        ) + c.sin_ts * (
            -c.cos_ti * zx - c.sin_ti * zx * zxp - c.sin_ti * zy * 
            zyp
        )
        c5 = -c.cos_ts * c.sin_ps * (
            -v * zx + v * zxp
        ) - c.cos_ps * c.cos_ts * (
            q + u * zxp + v * zy
        ) + c.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -c.cos_ts * c.sin_ps * (
            -u * zyp + q * zx * zyp) - c.cos_ps * c.cos_ts * (
            v * zyp - q * zy * zyp
        ) + c.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )

        rph = 1.0 + c.r_hh_i
        rmh = 1.0 - c.r_hh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)

        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)
        
        b1 = -c.cos_ts * c.sin_ps * (
            -1.0 - zx * zxp
        ) - c.sin_ts * zy - c.cos_ps * c.cos_ts * zxp * zy
        b2 = -c.cos_ts * c.sin_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ts * (
             -c.cos_ti * q * zy - q * c.sin_ti * zxp * zy + q *
             c.sin_ti * zx * zyp - c.cos_ti * u * zx * zyp -
             c.cos_ti * v * zy * zyp
        ) - c.cos_ps * c.cos_ts * (
             c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
             c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
             zy * zyp
        )
        b3 = -c.cos_ts * c.sin_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) - c.cos_ps * c.cos_ts * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        ) + c.sin_ts * (
             -c.sin_ti * v * zx + c.cos_ti * v * zx * zxp +
             c.sin_ti * u * zy - c.cos_ti * u * zxp * zy
        )
        b4 = -c.cos_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) + c.sin_ps * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        )
        b5 = -c.cos_ps * (-v * zx + v * zxp) + c.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -c.cos_ps * (-u * zyp + q * zx * zyp) + c.sin_ps * (
            v * zyp - q * zy * zyp
        )
        
        rp = 1.0 + c.r_vh_i
        rm = 1.0 - c.r_vh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)
        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)

        b1 = -c.cos_ts * c.sin_ps * (
            -1.0 - zx * zxp
        ) - c.sin_ts * zy - c.cos_ps * c.cos_ts * zxp * zy
        b2 = -c.cos_ts * c.sin_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp - c.sin_ti *
            u * zx * zxp - c.cos_ti * v * zyp - c.sin_ti * v * zx * zyp
        ) + c.sin_ts * (
             -c.cos_ti * q * zy - q * c.sin_ti * zxp * zy + q * c.sin_ti * zx *
             zyp - c.cos_ti * u * zx * zyp - c.cos_ti * v * zy * zyp
        ) - c.cos_ps * c.cos_ts * (
             c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q * c.sin_ti * zyp -
             c.cos_ti * u * zyp + c.sin_ti * v * zy * zyp
        )
        b3 = -c.cos_ts * c.sin_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp + c.cos_ti *
            q * zx * zxp
        ) - c.cos_ps * c.cos_ts * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        ) + c.sin_ts * (
            -c.sin_ti * v * zx + c.cos_ti * v * zx * zxp + c.sin_ti * u * zy -
            c.cos_ti * u * zxp * zy
        )
        b4 = -c.cos_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) + c.sin_ps * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        )
        b5 = -c.cos_ps * (-v * zx + v * zxp) + c.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -c.cos_ps * (
            -u * zyp + q * zx * zyp
        ) + c.sin_ps * (
            v * zyp - q * zy * zyp
        )

        rp = 1.0 + c.r_vh_i
        rm = 1.0 - c.r_vh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        kyv = v
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)
        
        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = kyv / (c.cos_ti + qslp)
        
        c1 = -c.cos_ps * (-1.0 - zx * zxp) + c.sin_ps * zxp * zy
        c2 = -c.cps_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ps * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
            zy * zyp
        )
        c3 = -c.cos_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) + c.sin_ps * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        )
        c4 = -c.cos_ts * c.sin_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        ) + c.sin_ts * (
            -c.cos_ti * zx - c.sin_ti * zx * zxp - c.sin_ti * zy *
            zyp
        )
        c5 = -c.cos_ts * c.sin_ps * (
            -v * zx + v * zxp
        ) - c.cos_ps * c.cos_ts * (
            q + u * zxp + v * zy
        ) + c.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -c.cos_ts * c.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            v * zyp - q * zy * zyp
        ) + c.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )
        rpv = 1.0 + c.r_vv_i
        rmv = 1.0 - c.r_vv_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)
        
        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)
        
        c1 = -c.cos_ps * (-1.0 - zx * zxp) + c.sin_ps * zxp * zy
        c2 = -c.cos_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ps * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
            zy * zyp
        )
        c3 = -c.cos_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp) + c.sin_ps * (
                -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti *
                zy - c.cos_ti * q * zxp * zy
        )
        c4 = -c.cos_ts * c.sin_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        ) + c.sin_ts * (
            -c.cos_ti * zx - c.sin_ti * zx * zxp - c.sin_ti * zy *
            zyp
        )
        c5 = -c.cos_ts * c.sin_ps * (
            -v * zx + v * zxp
        ) - c.cos_ps * c.cos_ts * (
            q + u * zxp + v * zy
        ) + c.sin_ts * (q * zx + u * zx * zxp + v * zxp * zy)
        c6 = -c.cos_ts * c.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            v * zyp - q * zy * zyp
        ) + c.sin_ts * (v * zx * zyp - u * zy * zyp)
        rph = 1.0 + c.r_hh_i
        rmh = 1.0 - c.r_hh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)
        
        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)
        
        b1 = -c.cos_ts * c.sin_ps * (
            -1.0 - zx * zxp
        ) - c.sin_ts * zy - c.cos_ps * c.cos_ts * zxp * zy
        b2 = -c.cos_ts * c.sin_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ts * (
            -c.cos_ti * q * zy - q * c.sin_ti * zxp * zy + q *
            c.sin_ti * zx * zyp - c.cos_ti * u * zx * zyp -
            c.cos_ti * v * zy * zyp
        ) - c.cos_ps * c.cos_ts * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
            zy * zyp
        )
        b3 = -c.cos_ts * c.sin_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) - c.cos_ps * c.cos_ts * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy) + c.sin_ts * (
            -c.sin_ti * v * zx + c.cos_ti * v * zx * zxp +
            c.sin_ti * u * zy - c.cos_ti * u * zxp * zy
        )
        b4 = -c.cos_ps * (-c.sin_ti * zyp + c.cos_ti * zx * zyp) + \
            c.sin_ps * (
                -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
             )
        b5 = -c.cos_ps * (-v * zx + v * zxp) + c.sin_ps * (
            q + u * zxp + v * zy
        )
        b6 = -c.cos_ps * (-u * zyp + q * zx * zyp) + c.sin_ps * (
            v * zyp - q * zy * zyp
        )
        rp = 1.0 + c.r_vh_i
        rm = 1.0 - c.r_vh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        v = v
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)

        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)

        b1 = -c.cos_ts * c.sin_ps * (
            -1.0 - zx * zxp
        ) - c.sin_ts * zy - c.cos_ps * c.cos_ts * zxp * zy
        b2 = -c.cos_ts * c.sin_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ts * (
            -c.cos_ti * q * zy - q * c.sin_ti * zxp * zy + q *
            c.sin_ti * zx * zyp - c.cos_ti * u * zx * zyp -
            c.cos_ti * v * zy * zyp
        ) - c.cos_ps * c.cos_ts * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
            zy * zyp
        )
        b3 = -c.cos_ts * c.sin_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) - c.cos_ps * c.cos_ts * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        ) + c.sin_ts * (
            -c.sin_ti * v * zx + c.cos_ti * v * zx * zxp +
            c.sin_ti * u * zy - c.cos_ti * u * zxp * zy
        )
        b4 = -c.cos_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) + c.sin_ps * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        )
        b5 = -c.cos_ps * (
            -v * zx + v * zxp
        ) + c.sin_ps * (q + u * zxp + v * zy)
        b6 = -c.cos_ps * (
            -u * zyp + q * zx * zyp
        ) + c.sin_ps * (v * zyp - q * zy * zyp)
        rp = 1.0 + c.r_vh_i
        rm = 1.0 - c.r_vh_i
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
        c = self.cache
        kxu = c.sin_ti + u
        ksxu = c.sin_ts * c.cos_ps + u
        ksyv = c.sin_ts * c.sin_ps + v
        if fabs((c.cos_ts - qslp).real) < precision:
            zx = 0.0
            zy = 0.0
        else:
            zx = (-ksxu) / (c.cos_ts - qslp)
            zy = (-ksyv) / (c.cos_ts - qslp)

        if fabs((c.cos_ti + qslp).real) < precision:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (c.cos_ti + qslp)
            zyp = v / (c.cos_ti + qslp)

        c1 = -c.cos_ps * (-1.0 - zx * zxp) + c.sin_ps * zxp * zy
        c2 = -c.cos_ps * (
            -c.cos_ti * q - c.cos_ti * u * zx - q * c.sin_ti * zxp -
            c.sin_ti * u * zx * zxp - c.cos_ti * v * zyp -
            c.sin_ti * v * zx * zyp
        ) + c.sin_ps * (
            c.cos_ti * u * zy + c.sin_ti * u * zxp * zy + q *
            c.sin_ti * zyp - c.cos_ti * u * zyp + c.sin_ti * v *
            zy * zyp
        )
        c3 = -c.cos_ps * (
            c.sin_ti * u - q * c.sin_ti * zx - c.cos_ti * u * zxp +
            c.cos_ti * q * zx * zxp
        ) + c.sin_ps * (
            -c.sin_ti * v + c.cos_ti * v * zxp + q * c.sin_ti * zy -
            c.cos_ti * q * zxp * zy
        )
        c4 = -c.cos_ts * c.sin_ps * (
            -c.sin_ti * zyp + c.cos_ti * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            -c.cos_ti - c.sin_ti * zxp - c.cos_ti * zy * zyp
        ) + c.sin_ts * (
            -c.cos_ti * zx - c.sin_ti * zx * zxp - c.sin_ti * zy *
            zyp
        )
        c5 = -c.cos_ts * c.sin_ps * (
            -v * zx + v * zxp
        ) - c.cos_ps * c.cos_ts * (
            q + u * zxp + v * zy
        ) + c.sin_ts * (
            q * zx + u * zx * zxp + v * zxp * zy
        )
        c6 = -c.cos_ts * c.sin_ps * (
            -u * zyp + q * zx * zyp
        ) - c.cos_ps * c.cos_ts * (
            v * zyp - q * zy * zyp
        ) + c.sin_ts * (
            v * zx * zyp - u * zy * zyp
        )
    
        rpv = 1.0 + c.r_vv_i
        rmv = 1.0 - c.r_vv_i
        av = rpv / qfix
        bv = rmv / qfix
        fbvvresult = av * (
            rpv * c1 - rmv * c2 - rpv * c3 / self.get_eps_r()
        ) - bv * (
            rmv * c4 * self.get_eps_r() + rpv * c5 + rmv * c6
        )
        return fbvvresult
