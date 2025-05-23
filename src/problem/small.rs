use crate::{
    alphabet::{small::InputAlphabet1, Alphabet},
    problem::Problem,
    symbol_relation::{
        small::{SmallSymbolRelation, SmallSymbolRelationAll},
        SymbolRelation,
    },
    symbol_set::small::SmallSymbolSet,
    util::{cartesian_product, CartesianProduct, Permutations},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmallProblem<AI: Alphabet, AO: Alphabet> {
    node_relation: SmallSymbolRelation<AI, AO>,
    edge_relation: SmallSymbolRelation<AO, AO>,
}

impl<AI: Alphabet, AO: Alphabet> SmallProblem<AI, AO> {
    pub fn new(
        node_relation: SmallSymbolRelation<AI, AO>,
        edge_relation: SmallSymbolRelation<AO, AO>,
    ) -> Self {
        Self {
            node_relation,
            edge_relation,
        }
    }

    pub fn all(input_alphabet: AI, output_alphabet: AO) -> SmallProblemIter<AI, AO> {
        SmallProblemIter(cartesian_product(
            SmallSymbolRelation::all(input_alphabet, output_alphabet),
            SmallSymbolRelation::all(output_alphabet, output_alphabet),
        ))
    }

    fn equivalent_problems(&self) -> EquivalentSmallProblemIter<'_, AI, AO> {
        EquivalentSmallProblemIter {
            problem: self,
            permutations: cartesian_product(
                Permutations::new(self.input_alphabet().size()),
                Permutations::new(self.output_alphabet().size()),
            ),
        }
    }

    fn permute_input(&self, perm: &[AI::Symbol]) -> Self {
        Self {
            node_relation: self.node_relation.permute_left(perm),
            edge_relation: self.edge_relation,
        }
    }

    fn permute_output(&self, perm: &[AO::Symbol]) -> Self {
        Self {
            node_relation: self.node_relation.permute_right(perm),
            edge_relation: self.edge_relation.permute_left(perm).permute_right(perm),
        }
    }
}

impl<AI: Alphabet, AO: Alphabet> Problem<AI, AO> for SmallProblem<AI, AO> {
    type NodeRelation = SmallSymbolRelation<AI, AO>;
    type EdgeRelation = SmallSymbolRelation<AO, AO>;
    type InputSymbolSet = SmallSymbolSet<AI>;
    type OutputSymbolSet = SmallSymbolSet<AO>;
    type InputRestriction = SmallProblem<AI::Smaller, AO>;
    type OutputRestriction = SmallProblem<AI, AO::Smaller>;

    fn restrict_input(&self, symbol: AI::Symbol) -> Self::InputRestriction {
        SmallProblem {
            node_relation: self.node_relation.restrict_left_symbol(symbol),
            edge_relation: self.edge_relation,
        }
    }

    fn restrict_output(&self, symbol: AO::Symbol) -> Self::OutputRestriction {
        SmallProblem {
            node_relation: self.node_relation.restrict_right_symbol(symbol),
            edge_relation: self
                .edge_relation
                .restrict_left_symbol(symbol)
                .restrict_right_symbol(symbol),
        }
    }

    fn fix_input_symbol(&self, symbol: AI::Symbol) -> SmallProblem<InputAlphabet1, AO> {
        SmallProblem {
            node_relation: self.node_relation.fix_left_symbol(symbol),
            edge_relation: self.edge_relation,
        }
    }

    fn remove_input_symbol(&self, symbol: AI::Symbol) -> Self {
        SmallProblem {
            node_relation: self.node_relation.remove_left_symbol(symbol),
            edge_relation: self.edge_relation,
        }
    }

    fn remove_output_symbol(&self, symbol: AO::Symbol) -> Self {
        SmallProblem {
            node_relation: self.node_relation.remove_right_symbol(symbol),
            edge_relation: self
                .edge_relation
                .remove_left_symbol(symbol)
                .remove_right_symbol(symbol),
        }
    }

    fn input_symbols(&self) -> Self::InputSymbolSet {
        self.node_relation.left_symbols()
    }

    fn output_symbols(&self) -> Self::OutputSymbolSet {
        self.node_relation.right_symbols()
            | self.edge_relation.left_symbols()
            | self.edge_relation.right_symbols()
    }

    fn node_relation(&self) -> &Self::NodeRelation {
        &self.node_relation
    }

    fn edge_relation(&self) -> &Self::EdgeRelation {
        &self.edge_relation
    }

    fn remove_from_edge_relation(&self, os1: AO::Symbol, os2: AO::Symbol) -> Self {
        SmallProblem {
            node_relation: self.node_relation,
            edge_relation: self.edge_relation.removed(os1, os2),
        }
    }

    fn input_alphabet(&self) -> AI {
        self.node_relation.left_alphabet()
    }

    fn output_alphabet(&self) -> AO {
        self.node_relation.right_alphabet()
    }

    fn normalize(&self) -> Self {
        self.equivalent_problems().min().unwrap()
    }
}

/// Iterator for constructing all problems.
///
/// See [`Problem::all`].
pub struct SmallProblemIter<AI: Alphabet, AO: Alphabet>(
    CartesianProduct<SmallSymbolRelationAll<AI, AO>, SmallSymbolRelationAll<AO, AO>>,
);

impl<AI: Alphabet, AO: Alphabet> Iterator for SmallProblemIter<AI, AO> {
    type Item = SmallProblem<AI, AO>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .next()
            .map(|(node_relation, edge_relation)| SmallProblem {
                node_relation,
                edge_relation,
            })
    }
}

pub struct EquivalentSmallProblemIter<'p, AI: Alphabet, AO: Alphabet> {
    problem: &'p SmallProblem<AI, AO>,
    permutations: CartesianProduct<Permutations, Permutations>,
}

impl<'p, AI: Alphabet, AO: Alphabet> Iterator for EquivalentSmallProblemIter<'p, AI, AO> {
    type Item = SmallProblem<AI, AO>;

    fn next(&mut self) -> Option<Self::Item> {
        let (input_perm, output_perm) = self.permutations.next()?;

        Some(
            self.problem
                .permute_input(
                    &input_perm
                        .into_iter()
                        .map(|s| self.problem.input_alphabet().symbol_from_index(s).unwrap())
                        .collect::<Vec<_>>(),
                )
                .permute_output(
                    &output_perm
                        .into_iter()
                        .map(|s| self.problem.output_alphabet().symbol_from_index(s).unwrap())
                        .collect::<Vec<_>>(),
                ),
        )
    }
}
