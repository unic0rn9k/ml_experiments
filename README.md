# Documentation
Documentation can be found in 2 different formats:
- Technical documentation, throught doc comments in the code. These can be viewed by running `cargo doc --open --all`
- Project overview/management, theory and thoughts behind the experiments. These can be found in the [journal](journal), which include the following files:
[5feb2024.md]( journal/5feb2024.md),
[10feb2024.md](journal/10feb2024.md),
[24feb2024.md](journal/24feb2024.md).

## CLI
- For a list of runnable experiments run `cargo r`
- To run one of them use `cargo r -- name_of_experiment`
- There are also experiments located in the crates folder. These can be listed with `ls crates` and run with `cargo r -p name_of_experiment`.
