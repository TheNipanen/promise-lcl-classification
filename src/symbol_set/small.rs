use std::{
    fmt,
    ops::{BitAnd, BitOr},
};

use super::{SymbolSet, SymbolSetIter};

use crate::alphabet::Alphabet;
use crate::alphabet::Symbol;

/// Set for containing symbols
///
/// Due to implementation reasons, the symbols contained in this set must come from an alphabet whose size is at most 8.
/// The constructor will panic if this is not the case.
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmallSymbolSet<A> {
    content: u8,
    alphabet: A,
}

impl<A: Alphabet> SmallSymbolSet<A> {
    /// Calculates the mask corresponding to the symbol.
    pub(crate) fn symbol_mask(symbol: A::Symbol) -> u8 {
        1 << symbol.index()
    }
}

impl<A: Alphabet> SymbolSet<A> for SmallSymbolSet<A> {
    type Iter = Self::IntoIter;

    /// Makes a new, empty [`SymbolSet`] for given alphabet.
    ///
    /// # Panic
    /// This function panics if the size of the alphabet is larger than 8.
    fn new(alphabet: A) -> Self {
        assert!(
            alphabet.size() <= 8,
            "SymbolSet cannot hold symbols from alphabet larger than 8."
        );
        Self {
            content: 0,
            alphabet,
        }
    }

    fn len(self) -> usize {
        self.content.count_ones() as usize
    }

    fn is_empty(self) -> bool {
        self.content == 0
    }

    fn insert(&mut self, symbol: A::Symbol) {
        self.content |= Self::symbol_mask(symbol);
    }

    fn remove(&mut self, symbol: A::Symbol) {
        self.content &= !Self::symbol_mask(symbol);
    }

    fn contains(&mut self, symbol: A::Symbol) -> bool {
        self.content & Self::symbol_mask(symbol) != 0
    }
}

impl<A: Alphabet> IntoIterator for SmallSymbolSet<A> {
    type Item = A::Symbol;

    type IntoIter = SmallSymbolSetIter<A>;

    fn into_iter(self) -> Self::IntoIter {
        SmallSymbolSetIter {
            inner: self,
            index: 0,
        }
    }
}

impl<A: Alphabet + Default> FromIterator<A::Symbol> for SmallSymbolSet<A> {
    fn from_iter<T: IntoIterator<Item = A::Symbol>>(iter: T) -> Self {
        let mut symbols = SmallSymbolSet::new(A::default());
        for symbol in iter {
            symbols.insert(symbol)
        }
        symbols
    }
}

impl<A: Alphabet> Extend<A::Symbol> for SmallSymbolSet<A> {
    fn extend<T: IntoIterator<Item = A::Symbol>>(&mut self, iter: T) {
        for symbol in iter {
            self.insert(symbol)
        }
    }
}

impl<A: Alphabet> BitAnd for SmallSymbolSet<A> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self {
            content: self.content & rhs.content,
            alphabet: self.alphabet,
        }
    }
}

impl<A: Alphabet> BitOr for SmallSymbolSet<A> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self {
            content: self.content | rhs.content,
            alphabet: self.alphabet,
        }
    }
}

impl<A: Alphabet> fmt::Debug for SmallSymbolSet<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(*self).finish()
    }
}

/// Iterator over all symbols in a [`SymbolSet`].
#[derive(Clone)]
pub struct SmallSymbolSetIter<A: Alphabet> {
    inner: SmallSymbolSet<A>,
    index: usize,
}

impl<A: Alphabet> SymbolSetIter<A> for SmallSymbolSetIter<A> {}

impl<A: Alphabet> Iterator for SmallSymbolSetIter<A> {
    type Item = A::Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let s = self.inner.alphabet.symbol_from_index(self.index)?;
            let result = self.inner.contains(s);
            self.index += 1;
            if result {
                return Some(s);
            }
        }
    }
}
