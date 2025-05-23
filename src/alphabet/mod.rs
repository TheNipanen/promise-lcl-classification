use std::{fmt, hash::Hash};

use crate::util;

pub mod small;

pub trait Alphabet: Copy + Ord + Hash + fmt::Debug {
    type Smaller: Alphabet;
    type Symbol: Symbol<Self>;
    type SymbolIter: Iterator<Item = Self::Symbol>;

    /// Returns the size of the alphabet.
    fn size(self) -> usize;

    /// Returns an alphabet that is one size smaller than the current one.
    fn smaller(self) -> Self::Smaller;

    /// Converts an index of a symbol to symbol.
    ///
    /// The symbol must be such that calling [`Symbol::index`] on it return back the same index.
    fn symbol_from_index(self, index: usize) -> Option<Self::Symbol>;

    /// Creates an iterator over all symbols in the alphabet.
    fn symbols(self) -> Self::SymbolIter;

    // Returns all symbol combinations of the given length
    fn combinations_of_len(self, length: u32) -> Vec<Vec<Self::Symbol>> {
        return util::combinations_of_len(&self.symbols().collect(), length);
    }
}

pub trait ConstAlphabet: Alphabet {
    const SIZE: usize;
}

pub trait Symbol<A: Alphabet>: Copy + Ord + Hash + fmt::Debug + fmt::Display {
    /// Returns the alphabet the symbol comes from.
    fn alphabet(self) -> A;

    /// Returns the index of the symbol in the alphabet.
    ///
    /// The index must be such that [`Alphabet::symbol_from_index`] returns back the same symbol.
    fn index(self) -> usize;
}

// pub struct AlphabetIter<A: Alphabet>(u16, A);

// impl<A: Alphabet> Iterator for AlphabetIter<A> {
//     type Item = Symbol<A>;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.0 >= self.1.size() {
//             return None;
//         }
//         let result = Symbol(self.0 as u8, PhantomData);
//         self.0 += 1;
//         Some(result)
//     }
// }

// #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
// pub struct Symbol<A: Alphabet>(A::Index, A);

// impl<A: Alphabet> Symbol<A> {
//     pub const fn new(symbol: u8, alphabet: A) -> Self {
//         assert!(
//             symbol < alphabet.size(),
//             "Symbol must come from the alphabet"
//         );
//         Self(symbol, alphabet)
//     }

//     pub fn alphabet(self) -> A {
//         self.1
//     }

//     pub fn index(self) -> A::Index {
//         self.0
//     }
// }

// impl<A: Alphabet> fmt::Debug for Symbol<A> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         self.1.display_fmt(self.0, f)
//     }
// }

// impl<A: Alphabet> fmt::Display for Symbol<A> {
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         self.1.display_fmt(self.0, f)
//     }
// }
