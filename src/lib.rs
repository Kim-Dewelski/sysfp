#![no_std]

#[cfg(target_arch = "x86_64")]
mod x86_imp {
    use core::arch::{asm, x86_64};

    #[repr(u32)]
    pub enum Rounding {
        /// Rounds towards zero.
        Zero = x86_64::_MM_ROUND_TOWARD_ZERO,
        /// Rounds towards positive infinity.
        Up = x86_64::_MM_ROUND_UP,
        /// Rounds towards negative infinity.
        Down = x86_64::_MM_ROUND_DOWN,
        /// Rounds towards nearest.
        Nearest = x86_64::_MM_ROUND_NEAREST,
    }

    /// The flags set for the operation.
    #[derive(Clone, Copy)]
    pub struct Flags {
        inner: u32,
    }

    impl Default for Flags {
        #[inline]
        fn default() -> Self {
            Self::new()
        }
    }

    impl Flags {
        #[inline]
        pub fn new() -> Self {
            Self {
                inner: x86_64::_MM_MASK_MASK,
            }
        }

        #[inline]
        pub fn with_rounding(mut self, rounding: Rounding) -> Self {
            self.set_rounding(rounding);
            self
        }

        #[inline]
        pub fn with_ftz(mut self, enabled: bool) -> Self {
            self.set_ftz(enabled);
            self
        }

        #[inline]
        pub fn set_rounding(&mut self, rounding: Rounding) {
            self.inner = (self.inner & !x86_64::_MM_ROUND_MASK) | rounding as u32;
        }

        #[inline]
        pub fn rounding(self) -> Rounding {
            match self.inner & x86_64::_MM_ROUND_MASK {
                b if b == Rounding::Zero as u32 => Rounding::Zero,
                b if b == Rounding::Up as u32 => Rounding::Up,
                b if b == Rounding::Down as u32 => Rounding::Down,
                _ => Rounding::Nearest,
            }
        }

        #[inline]
        pub fn set_ftz(&mut self, enabled: bool) {
            self.inner = (self.inner & !x86_64::_MM_FLUSH_ZERO_MASK)
                | if enabled {
                    x86_64::_MM_FLUSH_ZERO_ON
                } else {
                    x86_64::_MM_FLUSH_ZERO_OFF
                }
        }

        #[inline]
        pub fn ftz(self) -> bool {
            self.inner & x86_64::_MM_FLUSH_ZERO_MASK != 0
        }
    }

    /// The status from the operations.
    #[derive(Clone, Copy)]
    pub struct Status {
        inner: u32,
    }

    impl Status {
        pub const OVERFLOW: Self = Self {
            inner: x86_64::_MM_EXCEPT_OVERFLOW,
        };
        pub const UNDERFLOW: Self = Self {
            inner: x86_64::_MM_EXCEPT_UNDERFLOW,
        };
        pub const INEXACT: Self = Self {
            inner: x86_64::_MM_EXCEPT_INEXACT,
        };
        pub const DENORM: Self = Self {
            inner: x86_64::_MM_EXCEPT_DENORM,
        };
        pub const DIV_ZERO: Self = Self {
            inner: x86_64::_MM_EXCEPT_DIV_ZERO,
        };

        #[inline]
        pub fn empty() -> Self {
            Self { inner: 0 }
        }

        #[inline]
        pub fn has_exceptions(self) -> bool {
            self.inner & x86_64::_MM_EXCEPT_MASK != 0
        }

        #[inline]
        pub fn overflow(self) -> bool {
            self.has(Self::OVERFLOW)
        }

        #[inline]
        pub fn underflow(self) -> bool {
            self.has(Self::UNDERFLOW)
        }

        #[inline]
        pub fn inexact(self) -> bool {
            self.has(Self::INEXACT)
        }

        #[inline]
        pub fn denorm(self) -> bool {
            self.has(Self::DENORM)
        }

        #[inline]
        pub fn div_zero(self) -> bool {
            self.has(Self::DIV_ZERO)
        }

        #[inline]
        pub fn has(self, status: Self) -> bool {
            self.inner & status.inner == status.inner
        }

        #[inline]
        pub fn or(self, other: Self) -> Self {
            Self {
                inner: self.inner | other.inner,
            }
        }

        #[inline]
        pub fn and(self, other: Self) -> Self {
            Self {
                inner: self.inner & other.inner,
            }
        }
    }

    macro_rules! host_op {
        ($flags:ident; $asm:literal; $($end:tt)* ) => {
            unsafe {
                let mut status = 0;
                asm!(
                    "ldmxcsr [{flags:r}]",
                    $asm,
                    "stmxcsr [{status:r}]",
                    flags = in(reg) &$flags.inner as *const _,
                    status = in(reg) &mut status as *mut _,
                    $($end)*
                );
                status
            }
        };
    }

    pub mod f32 {
        // TODO: not yet implemented
    }

    pub mod f64 {
        use super::*;

        #[inline]
        pub fn add(flags: Flags, mut l: f64, r: f64) -> (f64, Status) {
            let status = host_op!(
                flags;
                "addsd {l}, {r}";
                l = inout(xmm_reg) l,
                r = in(xmm_reg) r
            );
            (l, Status { inner: status })
        }

        #[inline]
        pub fn sub(flags: Flags, mut l: f64, r: f64) -> (f64, Status) {
            let status = host_op!(
                flags;
                "subsd {l}, {r}";
                l = inout(xmm_reg) l,
                r = in(xmm_reg) r
            );
            (l, Status { inner: status })
        }

        #[inline]
        pub fn mul(flags: Flags, mut l: f64, r: f64) -> (f64, Status) {
            let status = host_op!(
                flags;
                "mulsd {l}, {r}";
                l = inout(xmm_reg) l,
                r = in(xmm_reg) r,
            );
            (l, Status { inner: status })
        }

        #[inline]
        pub fn div(flags: Flags, mut l: f64, r: f64) -> (f64, Status) {
            let status = host_op!(
                flags;
                "divsd {l}, {r}";
                l = inout(xmm_reg) l,
                r = in(xmm_reg) r,
            );
            (l, Status { inner: status })
        }

        #[inline]
        pub fn madd(flags: Flags, mut a: f64, b: f64, c: f64) -> (f64, Status) {
            let status = host_op!(
                flags;
                "vfmadd213sd {a}, {b}, {c}";
                a = inout(xmm_reg) a,
                b = in(xmm_reg) b,
                c = in(xmm_reg) c,
            );
            (a, Status { inner: status })
        }

        #[inline]
        pub fn to_single(flags: Flags, mut double: f64) -> (f32, Status) {
            let status = host_op!(
                flags;
                "cvtsd2ss {fp}, {fp}";
                fp = inout(xmm_reg) double,
            );
            (
                f32::from_bits(double.to_bits() as u32),
                Status { inner: status },
            )
        }
    }
}

cfg_if::cfg_if!(
    if #[cfg(target_arch = "x86_64")] {
        use x86_imp as imp;
    } else {
        mod empty {}

        static_assertions::const_assert!(false, "unsupported architecture for sysfp");
        use empty as imp;
    }
);

pub use imp::*;
