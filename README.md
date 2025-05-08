# GEO-Bench 2

![1-earth](https://github.com/The-AI-Alliance/GEO-Bench-2/assets/5478516/738b5aa6-b46d-48bc-bdde-fd71605b9bac)

## Installation

```shell
pip install -e .
```

## Unit Tests

Before we can use CI when the repo is public, we can still run unit tests to make sure things work as expected.

Pytests are contained in the tests directory and have configured path to the datasets that work atm, with paths on our toolkit workstation .

## Documentation

At the moment the documentation can only be built locally. For that you will have to install the optional-dependencies of the required documentation packages. You can do this locally from the project root with the command:

```shell
pip install -e ".[docs]"   
```

After that navigate to the `docs` folder in the command line and run `make clean` followed by `make html`. This will generate the documentation locally, which you can see in your browser by copying the html index link in the `_build` directory `docs/_build/html/index.html`.

## Generating the Benchmark

An underlying motivation of this effort is to reuse existing code and structures and only extend those existing frameworks for our purposes. This is why the dataset benchmark heavily relies on the [TorchGeo](https://github.com/microsoft/torchgeo) library for dataset loading and processing. 

The directory `./generate_benchmark` contains a script for each included dataset that has three purposes:

1. Generate a dataset subset that is sufficient for benchmark purposes and minimally in size to reduce disk memory
requirements for users
2. Generate possible partition sizes for experiments across dataset sizes
3. Generate a super tiny dataset version of dummy data that is used for unit testing all implemented functionality

## License

## Citation


