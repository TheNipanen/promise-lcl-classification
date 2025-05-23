use crate::{
    alphabet::Alphabet, classifier::ProblemClassificationInterpretation, problem::Problem,
    symbol_relation::SymbolRelation, symbol_set::SymbolSet, util::cartesian_product,
};

// Tests whether an algorithm mapping inputs of length <input_length - 1> to an output solves all allowed inputs of length <input_length>
pub fn test_algorithms<AI: Alphabet, AO: Alphabet>(
    problem: &impl Problem<AI, AO>,
    input_length: u32,
) -> Option<ProblemClassificationInterpretation> {
    let inputs: Vec<Vec<AI::Symbol>> = problem.inputs_of_len(input_length);
    let allowed_inputs: Vec<Vec<AI::Symbol>> = inputs
        .clone()
        .into_iter()
        .filter(|inp: &Vec<AI::Symbol>| input_is_allowed(problem, input_length, inp))
        .collect();

    let all_inputs_allowed = inputs.len() == allowed_inputs.len();

    let mapped_input_length = input_length - 1;

    for i in 0..mapped_input_length as usize {
        let mappings = problem.all_mappings(mapped_input_length, i);

        'algorithm_loop: for mapping in mappings {
            for inp in &allowed_inputs {
                let string1 = &inp[..(mapped_input_length as usize)].to_vec();
                let string2 = &inp[1..].to_vec();
                assert_eq!(string1.len(), mapped_input_length as usize);
                assert_eq!(string2.len(), mapped_input_length as usize);

                let is1 = string1[i];
                let is2 = string2[i];
                let os1 = mapping[string1];
                let os2 = mapping[string2];
                if !problem.node_relation().contains(is1, os1)
                    || !problem.node_relation().contains(is2, os2)
                    || !problem.edge_relation().contains(os1, os2)
                {
                    // Algorithm doesn't solve all allowed inputs
                    continue 'algorithm_loop;
                }
            }

            // All allowed inputs are solved by the mapping
            if all_inputs_allowed {
                return Some(ProblemClassificationInterpretation::AlwaysSolvableConstant);
            } else {
                return Some(ProblemClassificationInterpretation::SometimesSolvableConstant);
            }
        }
    }

    None
}

pub fn input_is_allowed<AI: Alphabet, AO: Alphabet>(
    problem: &impl Problem<AI, AO>,
    input_length: u32,
    inp: &Vec<AI::Symbol>,
) -> bool {
    let mut i = 1;
    let mut prev_limited_to = problem.node_relation().right_symbols_for(inp[0]);
    // Iterate all sequential pairs from input string
    while i < input_length {
        let is1 = inp[i as usize - 1];
        let is2 = inp[i as usize];
        let mut limit = prev_limited_to.clone();
        let output_pairs = cartesian_product(
            problem
                .node_relation()
                .right_symbols_for(is1)
                .into_iter()
                .filter(|os1| limit.contains(*os1)),
            problem.node_relation().right_symbols_for(is2).into_iter(),
        )
        .filter(|(os1, os2)| problem.edge_relation().contains(*os1, *os2))
        .collect::<Vec<_>>();
        // If any pair cannot be mapped to an output, then this input string is not allowed
        if output_pairs.len() == 0 {
            return false;
        }
        i += 1;
        prev_limited_to = SymbolSet::new(problem.output_alphabet());
        for (_os1, os2) in output_pairs {
            prev_limited_to.insert(os2);
        }
    }
    true
}

pub fn is_hardcoded_problem_1<AI: Alphabet, AO: Alphabet>(problem: &impl Problem<AI, AO>) -> bool {
    let inp = problem.input_alphabet();
    let out = problem.output_alphabet();

    let nodes = problem.node_relation();
    let edges = problem.edge_relation();

    if !(inp.size() == 2 && out.size() == 3 && nodes.len() == 4 && edges.len() == 6) {
        return false;
    }

    let ia = inp.symbol_from_index(0).unwrap();
    let ib = inp.symbol_from_index(1).unwrap();
    let oa = out.symbol_from_index(0).unwrap();
    let ob = out.symbol_from_index(1).unwrap();
    let oc = out.symbol_from_index(2).unwrap();

    [(ia, oa), (ia, oc), (ib, oa), (ib, ob)]
        .into_iter()
        .all(|(s1, s2)| nodes.contains(s1, s2))
        && [(oa, ob), (oa, oc), (ob, oa), (ob, ob), (oc, oa), (oc, oc)]
            .into_iter()
            .all(|(s1, s2)| edges.contains(s1, s2))
}

pub fn is_hardcoded_problem_2<AI: Alphabet, AO: Alphabet>(problem: &impl Problem<AI, AO>) -> bool {
    let inp = problem.input_alphabet();
    let out = problem.output_alphabet();

    let nodes = problem.node_relation();
    let edges = problem.edge_relation();

    if !(inp.size() == 2 && out.size() == 3 && nodes.len() == 5 && edges.len() == 4) {
        return false;
    }

    let ia = inp.symbol_from_index(0).unwrap();
    let ib = inp.symbol_from_index(1).unwrap();
    let oa = out.symbol_from_index(0).unwrap();
    let ob = out.symbol_from_index(1).unwrap();
    let oc = out.symbol_from_index(2).unwrap();

    [(ia, oa), (ia, ob), (ia, oc), (ib, oa), (ib, ob)]
        .into_iter()
        .all(|(s1, s2)| nodes.contains(s1, s2))
        && [(oa, ob), (ob, oc), (oc, oa), (oc, oc)]
            .into_iter()
            .all(|(s1, s2)| edges.contains(s1, s2))
}
