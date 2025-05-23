use std::{collections::HashSet, fmt, hash::Hash};

use crate::{
    alphabet::{small::InputAlphabet1, Alphabet},
    problem::small::SmallProblem,
    symbol_relation::SymbolRelation,
    symbol_set::SymbolSet,
    util,
};

pub mod small;

pub trait Problem<AI: Alphabet, AO: Alphabet>: Clone + Ord + Hash + fmt::Debug {
    type NodeRelation: SymbolRelation<
        AI,
        AO,
        LeftSymbolSet = Self::InputSymbolSet,
        RightSymbolSet = Self::OutputSymbolSet,
    >;
    type EdgeRelation: SymbolRelation<
        AO,
        AO,
        LeftSymbolSet = Self::OutputSymbolSet,
        RightSymbolSet = Self::OutputSymbolSet,
    >;
    type InputSymbolSet: SymbolSet<AI>;
    type OutputSymbolSet: SymbolSet<AO>;
    type InputRestriction: Problem<AI::Smaller, AO>;
    type OutputRestriction: Problem<AI, AO::Smaller>;

    fn restrict_input(&self, symbol: AI::Symbol) -> Self::InputRestriction;

    fn restrict_output(&self, symbol: AO::Symbol) -> Self::OutputRestriction;

    fn fix_input_symbol(&self, symbol: AI::Symbol) -> SmallProblem<InputAlphabet1, AO>;

    fn remove_input_symbol(&self, symbol: AI::Symbol) -> Self;

    fn remove_output_symbol(&self, symbol: AO::Symbol) -> Self;

    fn input_symbols(&self) -> Self::InputSymbolSet;

    fn output_symbols(&self) -> Self::OutputSymbolSet;

    fn node_relation(&self) -> &Self::NodeRelation;

    fn edge_relation(&self) -> &Self::EdgeRelation;

    fn remove_from_edge_relation(&self, os1: AO::Symbol, os2: AO::Symbol) -> Self;

    fn input_alphabet(&self) -> AI;

    fn output_alphabet(&self) -> AO;

    /// Returns the canonical version of the problem
    fn normalize(&self) -> Self;

    /// Returns an iterator for all possible mappings from input symbol string of given length to output symbol
    fn all_mappings(
        &self,
        input_length: u32,
        i: usize,
    ) -> util::AllMappings<Vec<AI::Symbol>, AO::Symbol> {
        let choices: Vec<(Vec<AI::Symbol>, Vec<AO::Symbol>)> = self
            .inputs_of_len(input_length)
            .into_iter()
            .map(|i_string| {
                (
                    i_string.clone(),
                    self.node_relation()
                        .right_symbols_for(i_string[i])
                        .into_iter()
                        .collect(),
                )
            })
            .collect();
        return util::all_mappings(choices);
    }

    // Returns all input symbol combinations of the given length
    fn inputs_of_len(&self, length: u32) -> Vec<Vec<AI::Symbol>> {
        return self.input_alphabet().combinations_of_len(length);
    }

    fn strongly_connected_output_components(&self) -> Vec<HashSet<AO::Symbol>> {
        return util::strongly_connected_components(self.edge_relation().clone().into_iter());
    }

    // Checks if an input symbol is compatible with at least one output of the given component, based on node relation
    fn input_compatible_with_some(&self, is: AI::Symbol, component: &HashSet<AO::Symbol>) -> bool {
        for os in component {
            if self.node_relation().contains(is, *os) {
                return true;
            }
        }
        false
    }

    // Checks if an input symbol is compatible with all outputs of the given component, based on node relation
    fn input_compatible_with_all(&self, is: AI::Symbol, component: &HashSet<AO::Symbol>) -> bool {
        for os in component {
            if !self.node_relation().contains(is, *os) {
                return false;
            }
        }
        true
    }
}
