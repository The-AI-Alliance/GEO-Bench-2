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

## Generating the Benchmark

An underlying motivation of this effort is to reuse existing code and structures and only extend those existing frameworks for our purposes. This is why the dataset benchmark heavily relies on the [TorchGeo](https://github.com/microsoft/torchgeo) library for dataset loading and processing. 

The directory `./generate_benchmark` contains a script for each included dataset that has three purposes:

1. Generate a dataset subset that is sufficient for benchmark purposes and minimally in size to reduce disk memory
requirements for users
2. Generate possible partition sizes for experiments across dataset sizes
3. Generate a super tiny dataset version of dummy data that is used for unit testing all implemented functionality

## License

## Citation


