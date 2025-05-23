use std::{collections::BTreeMap, fs::File, path::Path};

use serde::{ser::SerializeMap, Serialize, Serializer};

use crate::{
    alphabet::Alphabet,
    classifier::ProblemClassificationInterpretation,
    problem::{small::SmallProblem, Problem},
    symbol_relation::SymbolRelation,
};

impl<AI: Alphabet, AO: Alphabet> Serialize for SmallProblem<AI, AO> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let input_symbols = self.input_alphabet().symbols();
        let output_symbols = self.output_alphabet().symbols();
        let il = self.input_alphabet().size();
        let ol = self.output_alphabet().size();
        let mut result = String::with_capacity(il * (ol + 2) + ol * (ol + 2));
        for is in input_symbols {
            result.push_str(&format!(
                "{}:{} ",
                is,
                self.node_relation()
                    .right_symbols_for(is)
                    .into_iter()
                    .map(|os| os.to_string())
                    .collect::<String>()
            ));
        }
        for os1 in output_symbols {
            result.push_str(&format!(
                " {}:{}",
                os1,
                self.edge_relation()
                    .right_symbols_for(os1)
                    .into_iter()
                    .map(|os2| os2.to_string())
                    .collect::<String>()
            ));
        }
        serializer.serialize_str(&result)
    }
}

// Temporary wrapper struct to serialize the classifications for a problem
#[derive(Serialize)]
struct ClassificationState<'a> {
    classification: &'a Option<ProblemClassificationInterpretation>,
    count: &'a usize,
}

fn serialize_problems<AI: Alphabet, AO: Alphabet, S: Serializer>(
    unique_status: &BTreeMap<SmallProblem<AI, AO>, Option<ProblemClassificationInterpretation>>,
    equivalent_count: &BTreeMap<SmallProblem<AI, AO>, usize>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let length = unique_status.len();
    let mut map = serializer.serialize_map(Some(length))?;
    for problem in unique_status.keys() {
        map.serialize_entry(
            problem,
            &ClassificationState {
                classification: &unique_status[problem],
                count: &equivalent_count[problem],
            },
        )?;
    }

    return map.end();
}

pub fn save_classification<AI: Alphabet, AO: Alphabet>(
    unique_status: &BTreeMap<SmallProblem<AI, AO>, Option<ProblemClassificationInterpretation>>,
    equivalent_count: &BTreeMap<SmallProblem<AI, AO>, usize>,
    il: usize,
    ol: usize,
) {
    assert_eq!(unique_status.len(), equivalent_count.len());

    let path_string = format!("output/classification{}_{}.json", il, ol);
    let path = Path::new(&path_string);
    let display = path.display();

    let file = match File::create(path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };

    let mut serializer = serde_json::Serializer::pretty(file);

    match serialize_problems(unique_status, equivalent_count, &mut serializer) {
        Err(why) => panic!("couldn't write to {}: {}", display, why),
        Ok(_) => eprintln!("wrote to {}", display),
    }
}
