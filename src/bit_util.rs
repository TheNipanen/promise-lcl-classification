#[cfg(all(target_arch = "x86", target_feature = "bmi2"))]
mod x86 {
    use std::arch::x86;

    pub fn pdep_u32(a: u32, mask: u32) -> u32 {
        unsafe { _pdep_u32(a, mask) }
    }

    pub fn pext_u32(a: u32, mask: u32) -> u32 {
        unsafe { _pext_u32(a, mask) }
    }
}

#[cfg(all(target_arch = "x86", target_feature = "bmi2"))]
pub use x86::*;

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
mod x86_64 {
    use std::arch::x86_64;

    pub fn pdep_u32(a: u32, mask: u32) -> u32 {
        unsafe { _pdep_u32(a, mask) }
    }

    pub fn pext_u32(a: u32, mask: u32) -> u32 {
        unsafe { _pext_u32(a, mask) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
pub use x86_64::*;

#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "bmi2"
)))]
mod fallback {
    pub fn pdep_u32(a: u32, mask: u32) -> u32 {
        pdep_u32_impl(a, mask)
    }
    pub(super) fn pdep_u32_impl(a: u32, mask: u32) -> u32 {
        let mut dst = 0;
        let mut k = 0;
        for m in 0..32 {
            if mask & (1 << m) != 0 {
                if a & (1 << k) != 0 {
                    dst |= 1 << m;
                }
                k += 1;
            }
        }
        dst
    }

    pub fn pext_u32(a: u32, mask: u32) -> u32 {
        pext_u32_impl(a, mask)
    }

    pub(super) fn pext_u32_impl(a: u32, mask: u32) -> u32 {
        let mut dst = 0;
        let mut k = 0;
        for m in 0..32 {
            if mask & (1 << m) != 0 {
                if a & (1 << m) != 0 {
                    dst |= 1 << k;
                }
                k += 1;
            }
        }
        dst
    }
}

#[cfg(not(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "bmi2"
)))]
pub use fallback::*;

pub fn pdep_u16(a: u16, mask: u16) -> u16 {
    pdep_u32(a as u32, mask as u32) as u16
}

pub fn pext_u16(a: u16, mask: u16) -> u16 {
    pext_u32(a as u32, mask as u32) as u16
}

#[cfg(test)]
mod tests {
    use std::arch::x86_64::{_pdep_u32, _pext_u32};

    use crate::bit_util::fallback::{pdep_u32_impl, pext_u32_impl};

    #[test]
    fn test_pdep_u32_impl() {
        if is_x86_feature_detected!("bmi2") {
            assert_eq!(pdep_u32_impl(0, 0), unsafe { _pdep_u32(0, 0) });
            assert_eq!(pdep_u32_impl(0x123, 0x543), unsafe {
                _pdep_u32(0x123, 0x543)
            });
        } else {
            panic!("Cannot test against bmi2")
        }
    }

    #[test]
    fn test_pext_u32_impl() {
        if is_x86_feature_detected!("bmi2") {
            assert_eq!(pext_u32_impl(0, 0), unsafe { _pext_u32(0, 0) });
            assert_eq!(pext_u32_impl(0x123, 0x543), unsafe {
                _pext_u32(0x123, 0x543)
            });
        } else {
            panic!("Cannot test against bmi2")
        }
    }
}
