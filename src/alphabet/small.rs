//! Implementations of [`Alphabet`] for small, constant sizes.

use std::fmt;

use crate::alphabet::{Alphabet, ConstAlphabet, Symbol};

macro_rules! alphabet {
    ($name:ident, $size:literal, $smaller:ident, $alphabet:literal) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name;

        impl Alphabet for $name {
            type Smaller = $smaller;
            type Symbol = SmallSymbol<Self>;
            type SymbolIter = SmallAlphabetIter<Self>;

            fn size(self) -> usize {
                $size
            }

            fn smaller(self) -> Self::Smaller {
                $smaller
            }

            fn symbol_from_index(self, index: usize) -> Option<Self::Symbol> {
                #[allow(unused_comparisons)]
                if index < $size {
                    Some(SmallSymbol(index as u8, self))
                } else {
                    None
                }
            }

            fn symbols(self) -> Self::SymbolIter {
                SmallAlphabetIter(0, self)
            }
        }

        impl ConstAlphabet for $name {
            const SIZE: usize = $size;
        }

        impl Symbol<$name> for SmallSymbol<$name> {
            fn alphabet(self) -> $name {
                self.1
            }

            fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for SmallSymbol<$name> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                const ALPHABET: &str = $alphabet;
                let value = self.0 as usize;
                write!(f, "{}", &ALPHABET[value..=value])
            }
        }

        impl fmt::Debug for SmallSymbol<$name> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                const ALPHABET: &str = $alphabet;
                let value = self.0 as usize;
                write!(f, "{}", &ALPHABET[value..=value])
            }
        }
    };
}

macro_rules! input_alphabet {
    ($name:ident, $size:literal, $smaller:ident) => {
        alphabet!($name, $size, $smaller, "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    };
}

macro_rules! output_alphabet {
    ($name:ident, $size:literal, $smaller:ident) => {
        alphabet!($name, $size, $smaller, "abcdefghijklmnopqrstuvwxyz");
    };
}

input_alphabet!(InputAlphabet0, 0, OutputAlphabet0);
input_alphabet!(InputAlphabet1, 1, InputAlphabet0);
input_alphabet!(InputAlphabet2, 2, InputAlphabet1);
input_alphabet!(InputAlphabet3, 3, InputAlphabet2);
input_alphabet!(InputAlphabet4, 4, InputAlphabet3);

output_alphabet!(OutputAlphabet0, 0, OutputAlphabet0);
output_alphabet!(OutputAlphabet1, 1, OutputAlphabet0);
output_alphabet!(OutputAlphabet2, 2, OutputAlphabet1);
output_alphabet!(OutputAlphabet3, 3, OutputAlphabet2);
output_alphabet!(OutputAlphabet4, 4, OutputAlphabet3);

/// Concrete symbol for a small alphabet.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SmallSymbol<A>(u8, A);

impl<A: ConstAlphabet> SmallSymbol<A> {
    pub const fn new(index: usize, alphabet: A) -> Self {
        assert!(index < A::SIZE);
        Self(index as u8, alphabet)
    }
}

/// Iterator over all small symbols in an alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct SmallAlphabetIter<A>(usize, A);

impl<A: Alphabet<Symbol = SmallSymbol<A>>> Iterator for SmallAlphabetIter<A> {
    type Item = SmallSymbol<A>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 < self.1.size() {
            let result = self.0;
            self.0 += 1;
            return self.1.symbol_from_index(result);
        }
        None
    }
}
