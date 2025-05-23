# Classification of LCL promise problems

Source code located in `./src/`. 

Results located in `./output/`. Each JSON file contains classifications of all non-isomorphic problems in the given problem family, as well as the number of isomorphic problems for the given problem. The standard output of the classifier program is stored in `output.txt`, containing statistics about classified problems. 

To run the classifier, install the Rust package manager Cargo, then in the root folder run

`cargo run --release`
