use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt,
};

use serde::Serialize;

use crate::{
    alphabet::Alphabet,
    classifier_util::{
        input_is_allowed, is_hardcoded_problem_1, is_hardcoded_problem_2, test_algorithms,
    },
    problem::Problem,
    symbol_relation::SymbolRelation,
    symbol_set::SymbolSet,
    util::gcd,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum ProblemClassificationInterpretation {
    NeverSolvable,
    SometimesSolvableImpossible,
    SometimesSolvableLinear,
    SometimesSolvableConstant,
    AlwaysSolvableImpossible,
    AlwaysSolvableLinear,
    AlwaysSolvableConstant,
}

impl fmt::Display for ProblemClassificationInterpretation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NeverSolvable => f.pad("NeverSolvable"),
            Self::SometimesSolvableImpossible => f.pad("SometimesSolvableImpossible"),
            Self::SometimesSolvableLinear => f.pad("SometimesSolvableLinear"),
            Self::SometimesSolvableConstant => f.pad("SometimesSolvableConstant"),
            Self::AlwaysSolvableImpossible => f.pad("AlwaysSolvableImpossible"),
            Self::AlwaysSolvableLinear => f.pad("AlwaysSolvableLinear"),
            Self::AlwaysSolvableConstant => f.pad("AlwaysSolvableConstant"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProblemClassification {
    // Flags that can be pruned throughout the classifying process to reach a classification
    always_solvable: bool,    // Finite number of unsolvable inputs
    never_solvable: bool,     // Finite number of solvable inputs
    sometimes_solvable: bool, // Infinite number of both solvable and unsolvable inputs
    constant_locality: bool, // For a solvable input, the solution can be found by a PN algorithm with constant locality
    linear_locality: bool,
    impossible_locality: bool, // Solvable inputs cannot be solved by a PN algorithm
}

impl ProblemClassification {
    pub fn interpret(&self) -> Option<ProblemClassificationInterpretation> {
        // Check if the flags give a classification
        match (
            self.always_solvable,
            self.never_solvable,
            self.sometimes_solvable,
            self.constant_locality,
            self.linear_locality,
            self.impossible_locality,
        ) {
            (false, false, true, true, false, false) => {
                return Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
            }
            (false, false, true, false, true, false) => {
                return Some(ProblemClassificationInterpretation::SometimesSolvableLinear)
            }
            (false, false, true, false, false, true) => {
                return Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
            }
            (true, false, false, true, false, false) => {
                return Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
            }
            (true, false, false, false, true, false) => {
                return Some(ProblemClassificationInterpretation::AlwaysSolvableLinear)
            }
            (true, false, false, false, false, true) => {
                return Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible)
            }
            (false, true, false, _, _, _) => {
                return Some(ProblemClassificationInterpretation::NeverSolvable)
            }
            /* (true, true, _, _, _, _) => eprintln!("Problem is both always and never solvable"),
            (true, _, true, _, _, _) => eprintln!("Problem is both always and sometimes solvable"),
            (_, true, true, _, _, _) => eprintln!("Problem is both never and sometimes solvable"),
            (_, _, _, true, true, _) => eprintln!("Problem has constant and linear locality"),
            (_, _, _, true, _, true) => eprintln!("Problem has impossible and constant locality"),
            (_, _, _, _, true, true) => eprintln!("Problem has impossible and linear locality"), */
            (false, false, false, _, _, _) => {
                eprintln!("Problem is not never, always, or sometimes solvable")
            }
            (_, _, _, false, false, false) => {
                eprintln!("Problem is not impossible nor constant or linear")
            }
            _ => {}
        }

        // Cannot say anything about the problem 🤷
        None
    }

    fn set_classification(&mut self, classification: Option<ProblemClassificationInterpretation>) {
        (
            self.always_solvable,
            self.never_solvable,
            self.sometimes_solvable,
            self.constant_locality,
            self.linear_locality,
            self.impossible_locality,
        ) = match classification {
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant) => {
                (false, false, true, true, false, false)
            }
            Some(ProblemClassificationInterpretation::SometimesSolvableLinear) => {
                (false, false, true, false, true, false)
            }
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible) => {
                (false, false, true, false, false, true)
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant) => {
                (true, false, false, true, false, false)
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableLinear) => {
                (true, false, false, false, true, false)
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible) => {
                (true, false, false, false, false, true)
            }
            Some(ProblemClassificationInterpretation::NeverSolvable) => {
                (false, true, false, false, false, false)
            }
            None => (true, true, true, true, true, true),
        }
    }

    fn combine(&mut self, classification: Option<ProblemClassificationInterpretation>) {
        match classification {
            Some(ProblemClassificationInterpretation::NeverSolvable) => {
                self.always_solvable = false;
            }
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible) => {
                self.always_solvable = false;
                self.never_solvable = false;
                self.constant_locality = false;
                self.linear_locality = false;
            }
            Some(ProblemClassificationInterpretation::SometimesSolvableLinear) => {
                self.always_solvable = false;
                self.never_solvable = false;
                self.constant_locality = false;
            }
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant) => {
                self.always_solvable = false;
                self.never_solvable = false;
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible) => {
                self.never_solvable = false;
                self.constant_locality = false;
                self.linear_locality = false;
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableLinear) => {
                self.never_solvable = false;
                self.constant_locality = false;
            }
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant) => {
                self.never_solvable = false;
            }
            _ => {}
        }
    }
}

/// Classify the complexity of the problem.
pub fn classify<AI: Alphabet, AO: Alphabet>(
    problem: impl Problem<AI, AO>,
) -> ProblemClassification {
    let mut result = ProblemClassification {
        always_solvable: true,
        never_solvable: true,
        sometimes_solvable: true,
        constant_locality: true,
        linear_locality: true,
        impossible_locality: true,
    };

    // Check specific problems
    let normalized = problem.normalize();
    if is_hardcoded_problem_1(&normalized) {
        result.set_classification(Some(
            ProblemClassificationInterpretation::AlwaysSolvableImpossible,
        ));
        return result;
    }
    if is_hardcoded_problem_2(&normalized) {
        result.set_classification(Some(
            ProblemClassificationInterpretation::SometimesSolvableImpossible,
        ));
        return result;
    }

    if problem.input_symbols().is_empty() {
        // No input symbol can ever be present
        result.set_classification(Some(ProblemClassificationInterpretation::NeverSolvable));
        return result;
    }
    if problem.output_symbols().is_empty() {
        // No output symbol can ever be present
        result.set_classification(Some(ProblemClassificationInterpretation::NeverSolvable));
        return result;
    }

    // Prune output symbols which cannot be proceeded or succeeded by another output symbol
    for os in problem.output_symbols() {
        if !problem.edge_relation().left_symbols().contains(os) {
            // The symbol cannot be followed by anything
            return classify(problem.remove_output_symbol(os));
        }
        if !problem.edge_relation().right_symbols().contains(os) {
            // The symbol cannot be proceeded by anything
            return classify(problem.remove_output_symbol(os));
        }
    }

    // Prune output symbols which no input symbol can use
    for os in problem.output_symbols() {
        if !problem.node_relation().right_symbols().contains(os) {
            return classify(problem.remove_output_symbol(os));
        }
    }

    let components: Vec<HashSet<AO::Symbol>> = problem.strongly_connected_output_components();

    // Prune edges between different strongly connected components of edge relation
    for (os1, os2) in problem.edge_relation().clone() {
        let component1 = components
            .iter()
            .find(|set: &&HashSet<AO::Symbol>| set.contains(&os1));
        let component2 = components
            .iter()
            .find(|set: &&HashSet<AO::Symbol>| set.contains(&os2));
        if component1.unwrap() != component2.unwrap() {
            return classify(problem.remove_from_edge_relation(os1, os2));
        }
    }

    // Prune input symbols that are equivalent
    for is1 in problem.input_symbols() {
        for is2 in problem.input_symbols() {
            if is1 != is2
                && problem.node_relation().right_symbols_for(is1)
                    == problem.node_relation().right_symbols_for(is2)
            {
                return classify(problem.restrict_input(is2));
            }
        }
    }

    if problem.input_symbols().len() == 1 {
        // Unary input is the same as input-free case

        // Try to find loop state
        for os in problem.output_symbols() {
            if !problem.edge_relation().contains(os, os) {
                continue;
            }
            if !problem.node_relation().right_symbols().contains(os) {
                continue;
            }

            // The problem admits a constant solution
            if problem.input_alphabet().size() == 1 {
                result.set_classification(Some(
                    ProblemClassificationInterpretation::AlwaysSolvableConstant,
                ));
                return result;
            } else {
                result.set_classification(Some(
                    ProblemClassificationInterpretation::SometimesSolvableConstant,
                ));
                return result;
            }
        }

        // // Try to find a flexible state
        for os in problem.output_symbols() {
            let mut q = VecDeque::new();
            q.push_back((os, 0));
            let mut distances: HashMap<AO::Symbol, HashSet<usize>> = HashMap::new();
            while let Some((s, d)) = q.pop_front() {
                let dist = distances.entry(s).or_default();
                if !dist.insert(d) {
                    // There already exists a walk os -> s with length d
                    continue;
                }
                if d >= 2 * problem.output_alphabet().size() as usize {
                    continue;
                }
                for n in problem.edge_relation().right_symbols_for(s) {
                    q.push_back((n, d + 1));
                }
            }
            let gcd = distances[&os].iter().copied().reduce(gcd).unwrap_or(0);
            if gcd == 1 {
                // os is a flexible state

                if problem.input_alphabet().size() == 1 {
                    // The problem is always solvable, but PN cannot find that solution
                    result.set_classification(Some(
                        ProblemClassificationInterpretation::AlwaysSolvableImpossible,
                    ));
                    return result;
                } else {
                    // There might be some inputs for which the problem is not solvable
                    result.set_classification(Some(
                        ProblemClassificationInterpretation::SometimesSolvableImpossible,
                    ));
                    return result;
                }
            }
        }

        // The problem is very rigid, like 2-coloring or orientation. In particular, the length of the input cycle can be such that no input exists
        result.set_classification(Some(
            ProblemClassificationInterpretation::SometimesSolvableImpossible,
        ));
        return result;
    }

    // Check whether the problem is just about copying the input to output
    // Iterate all possible mappings from input to output and check if one works for all inputs of a given length
    for input_length in (2..5).step_by(2) {
        if input_length >= 4 && problem.input_alphabet().size() >= 4 {
            continue;
        }

        let algorithm_found = test_algorithms(&problem, input_length);
        if algorithm_found.is_some() {
            result.set_classification(algorithm_found);
            return result;
        }
    }

    // Check what we can deduce from the strongly connected components of the edge relation

    // If there is no component that is compatible with all input symbols, then problem cannot be always solvable
    let mut comp_component_found = false;
    'component_loop: for c in &components {
        for is in problem.input_alphabet().symbols() {
            if !problem.input_compatible_with_some(is, &c) {
                continue 'component_loop;
            }
        }
        comp_component_found = true;
    }
    if !comp_component_found {
        result.always_solvable = false;
    }

    // Check if there is no algorithm, proof based on symmetry breaking
    'symbol_loop: for is in problem.input_symbols() {
        let compatible_with_self_loops: Vec<AO::Symbol> = problem
            .node_relation()
            .right_symbols_for(is)
            .into_iter()
            .filter(|&os| problem.edge_relation().contains(os, os))
            .collect();
        let compatible_with_component = components
            .iter()
            .any(|component| problem.input_compatible_with_all(is, component));
        if compatible_with_self_loops.len() == 0 && compatible_with_component {
            // No self-loops, but still always solvable through full compatibility with a component
            // Then there is no algorithm
            result.constant_locality = false;
            result.linear_locality = false;
        } else if compatible_with_self_loops.len() > 0 {
            // If self-loops exist, then if for each self-loop, there is an input that can appear in a chain of 'is'
            // and is not compatible with any label in the self-loop's component, then there is no algorithm
            for self_loop in compatible_with_self_loops {
                let component = components
                    .iter()
                    .find(|component| component.contains(&self_loop))
                    .unwrap();
                let mut symbol_exists = false;
                for is2 in problem.input_symbols() {
                    let appears_in_chain = input_is_allowed(&problem, 3, &vec![is, is2, is]);
                    let is_compatible_with_component =
                        problem.input_compatible_with_some(is2, &component);
                    if appears_in_chain && !is_compatible_with_component {
                        symbol_exists = true;
                        break;
                    }
                }
                if !symbol_exists {
                    // No symbol exists for this self-loop, we can stop processing this iteration of symbol_loop
                    continue 'symbol_loop;
                }
            }
            // The described symbol exists for each self-loop, there is no algorithm
            result.constant_locality = false;
            result.linear_locality = false;
        }
    }

    // Check if we can deduce the solution from problem instances where we fix an input
    if problem.input_alphabet().size() != 1 {
        for is in problem.input_symbols() {
            let fixed_problem = problem.fix_input_symbol(is);
            let fixed_result = classify(fixed_problem);
            result.combine(fixed_result.interpret());
        }
    }

    // If we have no classification and the problem is not larger than 2,3-family, then we can try larger algorithms
    if problem.input_alphabet().size() <= 2
        && problem.output_alphabet().size() <= 3
        && result.interpret().is_none()
    {
        let algorithm_found = test_algorithms(&problem, 5);
        if algorithm_found.is_some() {
            result.set_classification(algorithm_found);
            return result;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::{
        alphabet::small::{
            InputAlphabet1, InputAlphabet2, InputAlphabet3, InputAlphabet4, OutputAlphabet1,
            OutputAlphabet2, OutputAlphabet3, OutputAlphabet4, SmallSymbol,
        },
        problem::small::SmallProblem,
    };

    use super::*;

    #[test]
    fn test_classification_interpretation() {
        const CLASSIFICATIONS: [Option<ProblemClassificationInterpretation>; 8] = [
            None,
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant),
            Some(ProblemClassificationInterpretation::AlwaysSolvableLinear),
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant),
            Some(ProblemClassificationInterpretation::SometimesSolvableLinear),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible),
            Some(ProblemClassificationInterpretation::NeverSolvable),
        ];

        let mut result = ProblemClassification {
            always_solvable: true,
            never_solvable: true,
            sometimes_solvable: true,
            constant_locality: true,
            linear_locality: true,
            impossible_locality: true,
        };
        for classification in CLASSIFICATIONS.into_iter() {
            result.set_classification(classification);
            assert_eq!(result.interpret(), classification);
        }
    }

    #[test]
    fn test_problem_classification_1_2() {
        type TestProblem = SmallProblem<InputAlphabet1, OutputAlphabet2>;
        const I: <InputAlphabet1 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet1);
        const A: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet2);
        const B: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet2);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        // 2-coloring
        assert_eq!(
            classify(TestProblem::new(
                [(I, A), (I, B)].into_iter().collect(),
                [(A, B), (B, A)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );
    }

    #[test]
    fn test_problem_classification_1_3() {
        type TestProblem = SmallProblem<InputAlphabet1, OutputAlphabet3>;
        const I: <InputAlphabet1 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet1);
        const A: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet3);
        const B: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet3);
        const C: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, OutputAlphabet3);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        // 2-coloring
        assert_eq!(
            classify(TestProblem::new(
                [(I, A), (I, B)].into_iter().collect(),
                [(A, B), (B, A)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // 3-coloring
        assert_eq!(
            classify(TestProblem::new(
                [(I, A), (I, B), (I, C)].into_iter().collect(),
                [(A, B), (A, C), (B, A), (B, C), (C, A), (C, B)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible)
        );
    }

    #[test]
    fn test_problem_classification_1_4() {
        type TestProblem = SmallProblem<InputAlphabet1, OutputAlphabet4>;
        const IA: <InputAlphabet1 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet1);
        const OA: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet4);
        const OB: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet4);
        const OC: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(2, OutputAlphabet4);
        const OD: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(3, OutputAlphabet4);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        // Sub-problems of color propagation
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA)].into_iter().collect(),
                [
                    (OA, OA),
                    (OA, OB),
                    (OB, OB),
                    (OB, OA),
                    (OA, OC),
                    (OC, OC),
                    (OC, OA),
                    (OC, OB),
                    (OB, OD),
                    (OD, OD),
                    (OD, OA),
                    (OD, OB)
                ]
                .into_iter()
                .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
        );
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OB)].into_iter().collect(),
                [
                    (OA, OA),
                    (OA, OB),
                    (OB, OB),
                    (OB, OA),
                    (OA, OC),
                    (OC, OC),
                    (OC, OA),
                    (OC, OB),
                    (OB, OD),
                    (OD, OD),
                    (OD, OA),
                    (OD, OB)
                ]
                .into_iter()
                .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
        );
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OC), (IA, OD)].into_iter().collect(),
                [
                    (OA, OA),
                    (OA, OB),
                    (OB, OB),
                    (OB, OA),
                    (OA, OC),
                    (OC, OC),
                    (OC, OA),
                    (OC, OB),
                    (OB, OD),
                    (OD, OD),
                    (OD, OA),
                    (OD, OB)
                ]
                .into_iter()
                .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
        );
    }

    #[test]
    fn test_problem_classification_2_2() {
        type TestProblem = SmallProblem<InputAlphabet2, OutputAlphabet2>;
        const IA: <InputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet2);
        const IB: <InputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(1, InputAlphabet2);
        const OA: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet2);
        const OB: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet2);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        // 2-coloring on unary alphabet should be sometimes solvable, but impossible in PN
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB)].into_iter().collect(),
                [(OA, OB), (OB, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // Same for regular 2-coloring
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OB, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // "Fill 2-coloring with occasional hints of the coloring"
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OB)].into_iter().collect(),
                [(OA, OB), (OB, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // "Fill 2-coloring with occasional hints of the coloring, variation"
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OA)].into_iter().collect(),
                [(OA, OB), (OB, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // "Copy input to output, but input B can occur only rarely. In addition, Bs can be outputted spuriously."
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OB)].into_iter().collect(),
                [(OA, OA), (OA, OB), (OB, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );
    }

    #[test]
    fn test_problem_classification_2_3() {
        type TestProblem = SmallProblem<InputAlphabet2, OutputAlphabet3>;
        const IA: <InputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet2);
        const IB: <InputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(1, InputAlphabet2);
        const OA: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet3);
        const OB: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet3);
        const OC: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, OutputAlphabet3);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OC)].into_iter().collect(),
                [(OA, OA), (OB, OC), (OC, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OC)].into_iter().collect(),
                [(OA, OA), (OB, OC), (OC, OA), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        // Some variations of a problem where each input has an output symbol that can always be outputted
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OB), (IA, OC), (IB, OA)].into_iter().collect(),
                [(OA, OA), (OA, OB), (OA, OC), (OB, OA), (OC, OA), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
        );
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OB), (IA, OC), (IB, OA)].into_iter().collect(),
                [(OA, OA), (OA, OB), (OA, OC), (OB, OA), (OB, OB), (OC, OA)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableConstant)
        );

        assert_eq!(
            classify(TestProblem::new(
                [(IA, OB), (IA, OC), (IB, OA)].into_iter().collect(),
                [(OA, OC), (OB, OB), (OC, OA)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IA, OC), (IB, OA)]
                    .into_iter()
                    .collect(),
                [(OA, OC), (OB, OA), (OB, OB), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IA, OC), (IB, OA)]
                    .into_iter()
                    .collect(),
                [(OA, OC), (OB, OA), (OB, OB), (OC, OC), (OB, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        // Case where non-symmetrical algorithm might be useful
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OC), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OB, OB), (OB, OC), (OC, OA)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        // Case where considering inputs of form (AB)^+ proves the classification
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OC), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OA, OC), (OB, OA), (OB, OB), (OC, OA), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible)
        );

        // Case where an algorithm for 4-length inputs should exist
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OC), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OB, OB), (OB, OC), (OC, OA), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        // Case where considering inputs of form (AAB)^+ proves the impossibility classification
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IA, OC), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OB, OC), (OC, OA), (OC, OC)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        // 4-length algorithm case that came up in a meeting
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OC), (IB, OA), (IB, OB)]
                    .into_iter()
                    .collect(),
                [(OA, OC), (OB, OA), (OB, OB), (OC, OB)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableConstant)
        );

        // Problem where we can prune input symbol
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IA, OC), (IB, OA), (IB, OB), (IB, OC)]
                    .into_iter()
                    .collect(),
                [(OA, OB), (OA, OC), (OB, OC), (OC, OA)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible)
        );
    }

    #[test]
    fn test_problem_classification_3_2() {
        type TestProblem = SmallProblem<InputAlphabet3, OutputAlphabet2>;
        const IA: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet3);
        const IB: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, InputAlphabet3);
        const IC: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, InputAlphabet3);
        const OA: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet2);
        const OB: <OutputAlphabet2 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet2);

        // Problems where we can deduce based on symmetry-breaking argument, that there is no algorithm
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OB), (IC, OA)]
                    .into_iter()
                    .collect(),
                [(OA, OA), (OB, OB)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );

        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IA, OB), (IB, OB), (IC, OA)]
                    .into_iter()
                    .collect(),
                [(OA, OA), (OA, OB), (OB, OB)].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );
    }

    #[test]
    fn test_problem_classification_3_3() {
        type TestProblem = SmallProblem<InputAlphabet3, OutputAlphabet3>;
        const IA: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet3);
        const IB: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, InputAlphabet3);
        const IC: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, InputAlphabet3);
        const OA: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet3);
        const OB: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet3);
        const OC: <OutputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, OutputAlphabet3);

        // Problem where we can prune input symbol
        assert_eq!(
            classify(TestProblem::new(
                [
                    (IA, OA),
                    (IA, OB),
                    (IA, OC),
                    (IB, OA),
                    (IB, OB),
                    (IB, OC),
                    (IC, OA),
                    (IC, OC)
                ]
                .into_iter()
                .collect(),
                [(OA, OB), (OA, OC), (OB, OC), (OC, OA)]
                    .into_iter()
                    .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::SometimesSolvableImpossible)
        );
    }

    #[test]
    fn test_problem_classification_3_4() {
        type TestProblem = SmallProblem<InputAlphabet3, OutputAlphabet4>;
        const IA: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(0, InputAlphabet3);
        const IB: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(1, InputAlphabet3);
        const IC: <InputAlphabet3 as Alphabet>::Symbol = SmallSymbol::new(2, InputAlphabet3);
        const OA: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(0, OutputAlphabet4);
        const OB: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(1, OutputAlphabet4);
        const OC: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(2, OutputAlphabet4);
        const OD: <OutputAlphabet4 as Alphabet>::Symbol = SmallSymbol::new(3, OutputAlphabet4);

        // Empty problem
        assert_eq!(
            classify(TestProblem::new(
                [].into_iter().collect(),
                [].into_iter().collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::NeverSolvable)
        );

        // Color propagation
        assert_eq!(
            classify(TestProblem::new(
                [(IA, OA), (IB, OB), (IC, OC), (IC, OD)]
                    .into_iter()
                    .collect(),
                [
                    (OA, OA),
                    (OA, OB),
                    (OB, OB),
                    (OB, OA),
                    (OA, OC),
                    (OC, OC),
                    (OC, OA),
                    (OC, OB),
                    (OB, OD),
                    (OD, OD),
                    (OD, OA),
                    (OD, OB)
                ]
                .into_iter()
                .collect()
            ))
            .interpret(),
            Some(ProblemClassificationInterpretation::AlwaysSolvableImpossible)
        );
    }

    #[test]
    fn test_equivalent_classification() {
        fn inner(input_alphabet: impl Alphabet, output_alphabet: impl Alphabet) {
            for problem in SmallProblem::all(input_alphabet, output_alphabet) {
                let normalized = problem.normalize();
                assert_eq!(
                    classify(problem).interpret(),
                    classify(normalized).interpret(),
                    "{problem:?} didn't classify same as its normal form {normalized:?}"
                );
            }
        }
        inner(InputAlphabet1, OutputAlphabet1);
        inner(InputAlphabet1, OutputAlphabet2);
        inner(InputAlphabet1, OutputAlphabet3);
        inner(InputAlphabet1, OutputAlphabet4);
        inner(InputAlphabet2, OutputAlphabet1);
        inner(InputAlphabet2, OutputAlphabet2);
        inner(InputAlphabet2, OutputAlphabet3);
        // inner(InputAlphabet2, OutputAlphabet4);
        inner(InputAlphabet3, OutputAlphabet1);
        inner(InputAlphabet3, OutputAlphabet2);
        // inner(InputAlphabet3, OutputAlphabet3);
        // inner(InputAlphabet3, OutputAlphabet4);
        inner(InputAlphabet4, OutputAlphabet1);
        inner(InputAlphabet4, OutputAlphabet2);
        // inner(InputAlphabet4, OutputAlphabet3);
        // inner(InputAlphabet4, OutputAlphabet4);
    }
}
