use std::{
    fmt,
    hash::Hash,
    ops::{BitAnd, BitOr},
};

use crate::alphabet::Alphabet;

pub mod small;

/// Set containing symbols of the given alphabet.
pub trait SymbolSet<A: Alphabet>:
    Clone
    + Ord
    + Hash
    + fmt::Debug
    + IntoIterator<Item = A::Symbol, IntoIter = Self::Iter>
    + Extend<A::Symbol>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Sized
{
    type Iter: SymbolSetIter<A>;

    /// Makes a new, empty [`SymbolSet`] for given alphabet.
    fn new(alphabet: A) -> Self;

    /// Returns the number of symbols contained in the set.
    fn len(self) -> usize;

    /// Checks whether the set is empty.
    fn is_empty(self) -> bool;

    /// Inserts a new symbol into the set.
    ///
    /// If the symbol is already contained in the set, this is a no-op.
    fn insert(&mut self, symbol: A::Symbol);

    /// Removes a symbol from the set.
    ///
    /// If the set does not contain the symbol, this is a no-op.
    fn remove(&mut self, symbol: A::Symbol);

    /// Checks whether the set contains the symbol.
    fn contains(&mut self, symbol: A::Symbol) -> bool;

    /// Computes the intersection of the set with another set.
    ///
    /// You can use the more succinct notation `left & right`.
    fn intersection(self, other: Self) -> Self {
        self & other
    }

    /// Computes the union of the set with another set.
    ///
    /// You can use the more succinct notation `left | right`.
    fn union(self, other: Self) -> Self {
        self | other
    }

    // /// Constructs a [`SymbolSet`] from an alphabet and iterator of symbols over that alphabet.
    // fn from_iter<T: IntoIterator<Item = A::Symbol>>(alphabet: A, iter: T) -> Self {
    //     let set = Self::new(alphabet);
    //     set.extend(iter);
    //     set
    // }
}

pub trait SymbolSetIter<A: Alphabet>: Clone + Iterator<Item = A::Symbol> {}
