use std::{fmt, hash::Hash};

use crate::{
    alphabet::{small::InputAlphabet1, Alphabet},
    symbol_relation::small::SmallSymbolRelation,
    symbol_set::SymbolSet,
};

pub mod small;

pub trait SymbolRelation<AL: Alphabet, AR: Alphabet>:
    Clone + Ord + Hash + fmt::Debug + IntoIterator<Item = (AL::Symbol, AR::Symbol)>
{
    type LeftSymbolSet: SymbolSet<AL>;
    type RightSymbolSet: SymbolSet<AR>;
    type LeftRestriction: SymbolRelation<AL::Smaller, AR>;
    type RightRestriction: SymbolRelation<AL, AR::Smaller>;

    /// Returns the number symbol pairs for which the relation holds.
    fn len(&self) -> usize;

    /// Checks whether the relation is empty.
    fn is_empty(&self) -> bool;

    /// Returns whether the relation contains the given symbol pair.
    fn contains(&self, s1: AL::Symbol, s2: AR::Symbol) -> bool;

    /// Adds a new pair to the relation.
    fn insert(&mut self, left: AL::Symbol, right: AR::Symbol);

    // Returns a copy of itself with the given pair removed
    fn removed(&self, left: AL::Symbol, right: AR::Symbol) -> Self;

    fn left_alphabet(&self) -> AL;

    fn right_alphabet(&self) -> AR;

    fn fix_left_symbol(&self, symbol: AL::Symbol) -> SmallSymbolRelation<InputAlphabet1, AR>;

    fn restrict_left_symbol(&self, symbol: AL::Symbol) -> Self::LeftRestriction;

    fn remove_left_symbol(&self, symbol: AL::Symbol) -> Self;

    fn restrict_right_symbol(&self, symbol: AR::Symbol) -> Self::RightRestriction;

    fn remove_right_symbol(&self, symbol: AR::Symbol) -> Self;

    fn left_symbols(&self) -> Self::LeftSymbolSet;

    fn left_symbols_for(&self, right: AR::Symbol) -> Self::LeftSymbolSet;

    fn right_symbols(&self) -> Self::RightSymbolSet;

    fn right_symbols_for(&self, left: AL::Symbol) -> Self::RightSymbolSet;

    fn permute_left(self, perm: &[AL::Symbol]) -> Self;

    fn permute_right(self, perm: &[AR::Symbol]) -> Self;
}
