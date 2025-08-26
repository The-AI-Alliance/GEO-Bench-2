# GEO-Bench 2

![1-earth](https://github.com/The-AI-Alliance/GEO-Bench-2/assets/5478516/738b5aa6-b46d-48bc-bdde-fd71605b9bac)

## Installation

```shell
pip install -e .
```

## Unit Tests

Before we can use CI when the repo is public, we can still run unit tests to make sure things work as expected.

Pytests are contained in the `tests` directory and are configured to use dataset paths that work at the moment, with paths on our toolkit workstation.

## Documentation

At the moment the documentation can only be built locally. To do so, install the optional dependency group for documentation from the project root with:

```shell
pip install -e ".[docs]"
```

After that, navigate to the `docs` folder and run `make clean` followed by `make html`. This will generate the documentation locally. Open `docs/_build/html/index.html` in your browser.

## Generating the Benchmark

An underlying motivation of this effort is to reuse existing code and structures and only extend those existing frameworks for our purposes. This is why the dataset benchmark heavily relies on the [TorchGeo](https://github.com/microsoft/torchgeo) library for dataset loading and processing. 

The directory `./generate_benchmark` contains a script for each included dataset that has three purposes:

1. Generate a dataset subset that is sufficient for benchmark purposes and minimal in size to reduce disk space
requirements for users.
2. Generate possible partition sizes for experiments across dataset sizes
3. Generate a super tiny dataset version of dummy data that is used for unit testing all implemented functionality

## License

## Citation


