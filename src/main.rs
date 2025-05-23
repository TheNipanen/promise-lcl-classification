use std::collections::BTreeMap;

use decider::{
    alphabet::{
        small::{
            InputAlphabet1, InputAlphabet2, InputAlphabet3, InputAlphabet4, OutputAlphabet1,
            OutputAlphabet2, OutputAlphabet3, OutputAlphabet4,
        },
        Alphabet,
    },
    classifier::{classify, ProblemClassificationInterpretation},
    file_util::save_classification,
    problem::{small::SmallProblem, Problem},
};

fn classify_and_summarize<AI: Alphabet, AO: Alphabet>(input_alphabet: AI, output_alphabet: AO) {
    println!(
        "Classifying {},{}-problems:",
        input_alphabet.size(),
        output_alphabet.size()
    );

    let mut summary: BTreeMap<ProblemClassificationInterpretation, usize> = BTreeMap::new();
    let mut unique_summary: BTreeMap<ProblemClassificationInterpretation, usize> = BTreeMap::new();
    let mut unique_status: BTreeMap<
        SmallProblem<AI, AO>,
        Option<ProblemClassificationInterpretation>,
    > = BTreeMap::new();
    // Count the number of equivalent problems for each canonical version of problem
    let mut equivalent_count: BTreeMap<SmallProblem<AI, AO>, usize> = BTreeMap::new();

    let mut classified_count = 0u64;
    let mut total_count = 0u64;

    let mut unique_classified = 0u64;
    let mut unique_total = 0u64;

    for problem in SmallProblem::all(input_alphabet, output_alphabet) {
        let classification_raw = classify(problem);
        let classification = classification_raw.interpret();

        total_count += 1;
        if let Some(classification) = classification {
            classified_count += 1;
            *summary.entry(classification).or_insert(0) += 1;
        }

        let normalized = problem.normalize();
        if normalized == problem {
            unique_total += 1;
            if let Some(classification) = classification {
                unique_classified += 1;
                *unique_summary.entry(classification).or_insert(0) += 1;
            }
        }

        *equivalent_count.entry(normalized).or_insert(0) += 1;

        if let Some(normal_classification) = unique_status.insert(normalized, classification) {
            assert_eq!(
                normal_classification, classification,
                "Normalization failed: {problem:?} {normalized:?}"
            );
        }

        if normalized == problem && classification.is_none() {
            eprintln!("Failed to classify problem {normalized:?} --- partial classification: {classification_raw:?}")
        } /* else if normalized == problem {
              eprintln!("Classified problem {normalized:?}: {classification:?}")
          } */
    }
    println!(
        "  Classified {classified_count} / {total_count} of {},{}-problems:",
        input_alphabet.size(),
        output_alphabet.size()
    );
    for (classification, count) in summary {
        println!("    {classification:30}: {count}");
    }
    if classified_count < total_count {
        println!(
            "    {:30}: {}",
            "Unclassified",
            total_count - classified_count
        );
    }
    println!("  Of those, {unique_classified} / {unique_total} were unique:");
    for (classification, count) in unique_summary {
        println!("    {classification:30}: {count}");
    }
    if unique_classified < unique_total {
        println!(
            "    {:30}: {}",
            "Unclassified",
            unique_total - unique_classified
        );
    }

    save_classification(
        &unique_status,
        &equivalent_count,
        input_alphabet.size(),
        output_alphabet.size(),
    );
}

fn main() {
    classify_and_summarize(InputAlphabet1, OutputAlphabet1);
    classify_and_summarize(InputAlphabet1, OutputAlphabet2);
    classify_and_summarize(InputAlphabet1, OutputAlphabet3);
    classify_and_summarize(InputAlphabet1, OutputAlphabet4);

    classify_and_summarize(InputAlphabet2, OutputAlphabet1);
    classify_and_summarize(InputAlphabet2, OutputAlphabet2);
    classify_and_summarize(InputAlphabet2, OutputAlphabet3);
    // classify_and_summarize(InputAlphabet2, OutputAlphabet4);

    classify_and_summarize(InputAlphabet3, OutputAlphabet1);
    classify_and_summarize(InputAlphabet3, OutputAlphabet2);
    // classify_and_summarize(InputAlphabet3, OutputAlphabet3);
    // classify_and_summarize(InputAlphabet3, OutputAlphabet4);

    classify_and_summarize(InputAlphabet4, OutputAlphabet1);
    classify_and_summarize(InputAlphabet4, OutputAlphabet2);
    // classify_and_summarize(InputAlphabet4, OutputAlphabet3);
    // classify_and_summarize(InputAlphabet4, OutputAlphabet4);
}
