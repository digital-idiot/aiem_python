from mpmath import mpmathify, mpf
from numpy import integer, floating

__all__ = ['EMWave']


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
