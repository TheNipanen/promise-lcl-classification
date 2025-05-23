use std::{fmt, hash::Hash};

use crate::{
    alphabet::{small::InputAlphabet1, Alphabet, Symbol},
    bit_util::{pdep_u16, pext_u16},
    symbol_relation::SymbolRelation,
    symbol_set::{small::SmallSymbolSet, SymbolSet},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmallSymbolRelation<AL: Alphabet, AR: Alphabet> {
    content: u16,
    left_alphabet: AL,
    right_alphabet: AR,
}

impl<AL: Alphabet, AR: Alphabet> SmallSymbolRelation<AL, AR> {
    pub fn all(left_alphabet: AL, right_alphabet: AR) -> SmallSymbolRelationAll<AL, AR> {
        assert!(
            left_alphabet.size().saturating_mul(right_alphabet.size()) <= 16,
            "The product of alphabet sizes must not exceed 16"
        );
        SmallSymbolRelationAll(0, left_alphabet, right_alphabet)
    }

    pub fn new(left_alphabet: AL, right_alphabet: AR) -> Self {
        Self {
            content: 0,
            left_alphabet,
            right_alphabet,
        }
    }

    fn symbol_mask(self, s1: AL::Symbol, s2: AR::Symbol) -> u16 {
        1 << (s1.index() * self.right_alphabet.size() + s2.index())
    }

    fn left_symbol_mask(self, s: AL::Symbol) -> u16 {
        let mask =
            ((1 << self.right_alphabet.size()) - 1) << (s.index() * self.right_alphabet.size());
        debug_assert_eq!(
            mask,
            self.right_alphabet
                .symbols()
                .map(|s2| self.symbol_mask(s, s2))
                .sum::<u16>()
        );
        mask
    }

    fn right_symbol_mask(self, s: AR::Symbol) -> u16 {
        // Construct mask for over all (s1, 0)
        let mask: u16 = (0..self.left_alphabet.size())
            .map(|i| 1 << (i * self.right_alphabet.size()))
            .sum();
        // Then shift if to correct position for s
        let mask = mask << s.index();
        debug_assert_eq!(
            mask,
            self.left_alphabet
                .symbols()
                .map(|s1| self.symbol_mask(s1, s))
                .sum::<u16>()
        );
        mask
    }
}

impl<AL: Alphabet, AR: Alphabet> SymbolRelation<AL, AR> for SmallSymbolRelation<AL, AR> {
    type LeftSymbolSet = SmallSymbolSet<AL>;
    type RightSymbolSet = SmallSymbolSet<AR>;
    type LeftRestriction = SmallSymbolRelation<AL::Smaller, AR>;
    type RightRestriction = SmallSymbolRelation<AL, AR::Smaller>;

    /// Returns the number symbol pairs for which the relation holds.
    fn len(&self) -> usize {
        self.content.count_ones() as usize
    }

    /// Checks whether the relation is empty.
    fn is_empty(&self) -> bool {
        self.content == 0
    }

    /// Returns whether the relation contains the given symbol pair.
    fn contains(&self, s1: AL::Symbol, s2: AR::Symbol) -> bool {
        self.content & self.symbol_mask(s1, s2) != 0
    }

    fn insert(&mut self, left: AL::Symbol, right: AR::Symbol) {
        self.content |= self.symbol_mask(left, right)
    }

    fn removed(&self, left: <AL as Alphabet>::Symbol, right: <AR as Alphabet>::Symbol) -> Self {
        SmallSymbolRelation {
            content: self.content & !self.symbol_mask(left, right),
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet,
        }
    }

    fn left_alphabet(&self) -> AL {
        self.left_alphabet
    }

    fn right_alphabet(&self) -> AR {
        self.right_alphabet
    }

    fn fix_left_symbol(&self, symbol: AL::Symbol) -> SmallSymbolRelation<InputAlphabet1, AR> {
        let mask = self.left_symbol_mask(symbol);
        // Shift the content, renaming the symbol as the first symbol of the new unary alphabet
        let shifted_content = (self.content & mask) >> mask.trailing_zeros();
        SmallSymbolRelation {
            content: shifted_content,
            left_alphabet: InputAlphabet1,
            right_alphabet: self.right_alphabet,
        }
    }

    fn restrict_left_symbol(&self, symbol: AL::Symbol) -> Self::LeftRestriction {
        let mask = self.left_symbol_mask(symbol);
        SmallSymbolRelation {
            content: pext_u16(self.content, !mask),
            left_alphabet: self.left_alphabet.smaller(),
            right_alphabet: self.right_alphabet,
        }
    }

    fn remove_left_symbol(&self, symbol: AL::Symbol) -> Self {
        // Zero out removed bits
        Self {
            content: self.content & !self.left_symbol_mask(symbol),
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet,
        }
    }

    fn restrict_right_symbol(&self, symbol: AR::Symbol) -> Self::RightRestriction {
        let mask = self.right_symbol_mask(symbol);
        SmallSymbolRelation {
            content: pext_u16(self.content, !mask),
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet.smaller(),
        }
    }

    fn remove_right_symbol(&self, symbol: AR::Symbol) -> Self {
        Self {
            content: self.content & !self.right_symbol_mask(symbol),
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet,
        }
    }

    fn left_symbols(&self) -> Self::LeftSymbolSet {
        let mut set = SmallSymbolSet::new(self.left_alphabet);
        set.extend(self.into_iter().map(|(left, _right)| left));
        set
    }

    fn left_symbols_for(&self, right: AR::Symbol) -> Self::LeftSymbolSet {
        let mut set = SmallSymbolSet::new(self.left_alphabet);
        set.extend(
            self.into_iter()
                .filter(|(_, r)| *r == right)
                .map(|(l, _)| l),
        );
        set
    }

    fn right_symbols(&self) -> Self::RightSymbolSet {
        let mut set = SmallSymbolSet::new(self.right_alphabet);
        set.extend(self.into_iter().map(|(_left, right)| right));
        set
    }

    fn right_symbols_for(&self, left: AL::Symbol) -> Self::RightSymbolSet {
        let mut set = SmallSymbolSet::new(self.right_alphabet);
        set.extend(self.into_iter().filter(|(l, _)| *l == left).map(|(_, r)| r));
        set
    }

    fn permute_left(self, perm: &[AL::Symbol]) -> Self {
        assert_eq!(perm.len(), self.left_alphabet.size());
        let mut new = 0;

        for (i, s) in perm.into_iter().copied().enumerate() {
            // Extract symbol according to the permutation and then place it as ith symbol
            let bits = pext_u16(self.content, self.left_symbol_mask(s));
            new |= pdep_u16(
                bits,
                self.left_symbol_mask(self.left_alphabet.symbol_from_index(i).unwrap()),
            );
        }

        Self {
            content: new,
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet,
        }
    }

    fn permute_right(self, perm: &[AR::Symbol]) -> Self {
        assert_eq!(perm.len(), self.right_alphabet.size());
        let mut new = 0;

        for (i, s) in perm.into_iter().copied().enumerate() {
            // Extract symbol according to the permutation and then place it as ith symbol
            let bits = pext_u16(self.content, self.right_symbol_mask(s));
            new |= pdep_u16(
                bits,
                self.right_symbol_mask(self.right_alphabet.symbol_from_index(i).unwrap()),
            );
        }

        Self {
            content: new,
            left_alphabet: self.left_alphabet,
            right_alphabet: self.right_alphabet,
        }
    }
}

impl<AL: Alphabet, AR: Alphabet> fmt::Debug for SmallSymbolRelation<AL, AR> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(*self).finish()
    }
}

// impl<AL: Alphabet, AR: Alphabet> Clone for SmallSymbolRelation<AL, AR> {
//     fn clone(&self) -> Self {

//         Self(self.content.clone(), self.1.clone())
//     }
// }
// impl<AL: Alphabet, AR: Alphabet> Copy for SmallSymbolRelation<AL, AR> {}
// impl<AL: Alphabet, AR: Alphabet> PartialEq for SmallSymbolRelation<AL, AR> {
//     fn eq(&self, other: &Self) -> bool {
//         self.content == other.0 && self.1 == other.1
//     }
// }
// impl<AL: Alphabet, AR: Alphabet> Eq for SmallSymbolRelation<AL, AR> {}
// impl<AL: Alphabet, AR: Alphabet> PartialOrd for SmallSymbolRelation<AL, AR> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.content.partial_cmp(&other.0)
//     }
// }
// impl<AL: Alphabet, AR: Alphabet> Ord for SmallSymbolRelation<AL, AR> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.content.cmp(&other.0)
//     }
// }
// impl<AL: Alphabet, AR: Alphabet> Hash for SmallSymbolRelation<AL, AR> {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.content.hash(state);
//         self.1.hash(state);
//     }
// }

impl<AL: Alphabet, AR: Alphabet> IntoIterator for SmallSymbolRelation<AL, AR> {
    type Item = (AL::Symbol, AR::Symbol);

    type IntoIter = SymbolRelationIter<AL, AR>;

    fn into_iter(self) -> Self::IntoIter {
        SymbolRelationIter(self, 0)
    }
}

impl<AL: Alphabet + Default, AR: Alphabet + Default> FromIterator<(AL::Symbol, AR::Symbol)>
    for SmallSymbolRelation<AL, AR>
{
    fn from_iter<T: IntoIterator<Item = (AL::Symbol, AR::Symbol)>>(iter: T) -> Self {
        let mut relation = Self::new(AL::default(), AR::default());
        relation.extend(iter);
        relation
    }
}

impl<AL: Alphabet, AR: Alphabet> Extend<(AL::Symbol, AR::Symbol)> for SmallSymbolRelation<AL, AR> {
    fn extend<T: IntoIterator<Item = (AL::Symbol, AR::Symbol)>>(&mut self, iter: T) {
        for (s1, s2) in iter {
            self.insert(s1, s2);
        }
    }
}

pub struct SymbolRelationIter<AL: Alphabet, AR: Alphabet>(SmallSymbolRelation<AL, AR>, u8);

impl<AL: Alphabet, AR: Alphabet> Iterator for SymbolRelationIter<AL, AR> {
    type Item = (AL::Symbol, AR::Symbol);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.1 as usize >= self.0.left_alphabet.size() * self.0.right_alphabet.size() {
                return None;
            }

            let s1 = self
                .0
                .left_alphabet
                .symbol_from_index(self.1 as usize / self.0.right_alphabet.size())?;
            let s2 = self
                .0
                .right_alphabet
                .symbol_from_index(self.1 as usize % self.0.right_alphabet.size())?;
            debug_assert!(
                s1.index() * self.0.right_alphabet.size() + s2.index() == self.1 as usize
            );
            self.1 += 1;
            if self.0.contains(s1, s2) {
                return Some((s1, s2));
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SmallSymbolRelationAll<AL: Alphabet, AR: Alphabet>(u32, AL, AR);

impl<AL: Alphabet, AR: Alphabet> Iterator for SmallSymbolRelationAll<AL, AR> {
    type Item = SmallSymbolRelation<AL, AR>;

    fn next(&mut self) -> Option<Self::Item> {
        debug_assert!(
            self.1.size().saturating_mul(self.2.size()) <= 16,
            "The product of alphabet sizes must not exceed 16"
        );
        if self.0 >= 1 << (self.1.size() * self.2.size()) {
            return None;
        }
        let result = SmallSymbolRelation {
            content: self.0 as u16,
            left_alphabet: self.1,
            right_alphabet: self.2,
        };
        self.0 += 1;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::alphabet::small::{OutputAlphabet2, OutputAlphabet3, OutputAlphabet4, SmallSymbol};

    type TestAlphabet2 = OutputAlphabet2;
    type TestAlphabet3 = OutputAlphabet3;
    type TestAlphabet4 = OutputAlphabet4;
    const A2A: SmallSymbol<TestAlphabet2> = SmallSymbol::new(0, OutputAlphabet2);
    const A2B: SmallSymbol<TestAlphabet2> = SmallSymbol::new(1, OutputAlphabet2);
    const A3A: SmallSymbol<TestAlphabet3> = SmallSymbol::new(0, OutputAlphabet3);
    const A3B: SmallSymbol<TestAlphabet3> = SmallSymbol::new(1, OutputAlphabet3);
    const A3C: SmallSymbol<TestAlphabet3> = SmallSymbol::new(2, OutputAlphabet3);
    const A4A: SmallSymbol<TestAlphabet4> = SmallSymbol::new(0, OutputAlphabet4);
    const A4B: SmallSymbol<TestAlphabet4> = SmallSymbol::new(1, OutputAlphabet4);
    const A4C: SmallSymbol<TestAlphabet4> = SmallSymbol::new(2, OutputAlphabet4);
    const A4D: SmallSymbol<TestAlphabet4> = SmallSymbol::new(3, OutputAlphabet4);

    use super::*;

    #[test]
    fn test_from_iterator() {
        let relation: SmallSymbolRelation<TestAlphabet2, TestAlphabet2> =
            [(A2A, A2A), (A2A, A2B)].into_iter().collect();
        assert_eq!(relation.len(), 2);
        assert!(!relation.is_empty());
        assert!(relation.contains(A2A, A2A));
        assert!(relation.contains(A2A, A2B));
        assert!(!relation.contains(A2B, A2A));
        assert!(!relation.contains(A2B, A2B));

        let relation: SmallSymbolRelation<TestAlphabet2, TestAlphabet2> = [].into_iter().collect();
        assert_eq!(relation.len(), 0);
        assert!(relation.is_empty());
        assert!(!relation.contains(A2A, A2A));
        assert!(!relation.contains(A2A, A2B));
        assert!(!relation.contains(A2B, A2A));
        assert!(!relation.contains(A2B, A2B));

        let relation: SmallSymbolRelation<TestAlphabet2, TestAlphabet3> =
            [(A2B, A3A), (A2A, A3C), (A2A, A3A), (A2B, A3A)]
                .into_iter()
                .collect();
        assert_eq!(relation.len(), 3);
        assert!(!relation.is_empty());
        assert!(relation.contains(A2A, A3A));
        assert!(!relation.contains(A2A, A3B));
        assert!(relation.contains(A2A, A3C));
        assert!(relation.contains(A2B, A3A));
        assert!(!relation.contains(A2B, A3B));
        assert!(!relation.contains(A2B, A3C));
    }

    #[test]
    fn test_remove_left_symbol() {
        let relation: SmallSymbolRelation<TestAlphabet4, TestAlphabet4> = [
            (A4A, A4A),
            (A4A, A4B),
            (A4A, A4C),
            (A4A, A4D),
            (A4B, A4A),
            (A4B, A4D),
            (A4C, A4A),
            (A4C, A4B),
            (A4C, A4C),
            (A4D, A4A),
            (A4D, A4D),
        ]
        .into_iter()
        .collect();

        let relation_la = relation.remove_left_symbol(A4A);
        assert!(!relation_la.contains(A4A, A4A));
        assert!(!relation_la.contains(A4A, A4B));
        assert!(!relation_la.contains(A4A, A4C));
        assert!(!relation_la.contains(A4A, A4D));
        assert!(relation_la.contains(A4B, A4A));
        assert!(relation_la.contains(A4B, A4D));
        assert!(relation_la.contains(A4C, A4A));
        assert!(relation_la.contains(A4C, A4B));
        assert!(relation_la.contains(A4C, A4C));
        assert!(relation_la.contains(A4D, A4A));
        assert!(relation_la.contains(A4D, A4D));
    }

    #[test]
    fn test_restrict_left_symbol() {
        let relation: SmallSymbolRelation<TestAlphabet4, TestAlphabet4> = [
            (A4A, A4A),
            (A4A, A4B),
            (A4A, A4C),
            (A4A, A4D),
            (A4B, A4A),
            (A4B, A4D),
            (A4C, A4A),
            (A4C, A4B),
            (A4C, A4C),
            (A4D, A4A),
            (A4D, A4D),
        ]
        .into_iter()
        .collect();

        let relation_la: SmallSymbolRelation<TestAlphabet3, TestAlphabet4> =
            relation.restrict_left_symbol(A4A);

        assert!(relation_la.contains(A3A, A4A));
        assert!(!relation_la.contains(A3A, A4B));
        assert!(!relation_la.contains(A3A, A4C));
        assert!(relation_la.contains(A3A, A4D));

        assert!(relation_la.contains(A3B, A4A));
        assert!(relation_la.contains(A3B, A4B));
        assert!(relation_la.contains(A3B, A4C));
        assert!(!relation_la.contains(A3B, A4D));

        assert!(relation_la.contains(A3C, A4A));
        assert!(!relation_la.contains(A3C, A4B));
        assert!(!relation_la.contains(A3C, A4C));
        assert!(relation_la.contains(A3C, A4D));
    }

    #[test]
    fn test_remove_right_symbol() {
        let relation: SmallSymbolRelation<TestAlphabet4, TestAlphabet4> = [
            (A4A, A4A),
            (A4A, A4B),
            (A4A, A4C),
            (A4A, A4D),
            (A4B, A4A),
            (A4B, A4D),
            (A4C, A4A),
            (A4C, A4B),
            (A4C, A4C),
            (A4D, A4A),
            (A4D, A4D),
        ]
        .into_iter()
        .collect();

        let relation_la = relation.remove_right_symbol(A4A);
        assert!(!relation_la.contains(A4A, A4A));
        assert!(relation_la.contains(A4A, A4B));
        assert!(relation_la.contains(A4A, A4C));
        assert!(relation_la.contains(A4A, A4D));
        assert!(!relation_la.contains(A4B, A4A));
        assert!(!relation_la.contains(A4B, A4B));
        assert!(!relation_la.contains(A4B, A4C));
        assert!(relation_la.contains(A4B, A4D));
        assert!(!relation_la.contains(A4C, A4A));
        assert!(relation_la.contains(A4C, A4B));
        assert!(relation_la.contains(A4C, A4C));
        assert!(!relation_la.contains(A4C, A4D));
        assert!(!relation_la.contains(A4D, A4A));
        assert!(!relation_la.contains(A4D, A4B));
        assert!(!relation_la.contains(A4D, A4C));
        assert!(relation_la.contains(A4D, A4D));
    }

    #[test]
    fn test_restrict_right_symbol() {
        let relation: SmallSymbolRelation<TestAlphabet4, TestAlphabet4> = [
            (A4A, A4A),
            (A4A, A4B),
            (A4A, A4C),
            (A4A, A4D),
            (A4B, A4A),
            (A4B, A4D),
            (A4C, A4A),
            (A4C, A4B),
            (A4C, A4C),
            (A4D, A4A),
            (A4D, A4D),
        ]
        .into_iter()
        .collect();

        let relation_la: SmallSymbolRelation<TestAlphabet4, TestAlphabet3> =
            relation.restrict_right_symbol(A4A);

        assert!(relation_la.contains(A4A, A3A));
        assert!(relation_la.contains(A4A, A3B));
        assert!(relation_la.contains(A4A, A3C));
        assert!(!relation_la.contains(A4B, A3A));
        assert!(!relation_la.contains(A4B, A3B));
        assert!(relation_la.contains(A4B, A3C));
        assert!(relation_la.contains(A4C, A3A));
        assert!(relation_la.contains(A4C, A3B));
        assert!(!relation_la.contains(A4C, A3C));
        assert!(!relation_la.contains(A4D, A3A));
        assert!(!relation_la.contains(A4D, A3B));
        assert!(relation_la.contains(A4D, A3C));
    }
}
