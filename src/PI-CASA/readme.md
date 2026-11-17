# Running PI-CASA

Run the following commands from this directory. Linux, GNU make, and g++ are required. The included `PI-CASA` binary is built for Linux x86-64.


## Run a Single Benchmark

2-way example:

```bash
mkdir -p output/2way
./PI-CASA -s 1 -y 1000 \
  -c src/PI-CASA/2_way_model/benchmark_1.constraints \
  -o output/2way/result.txt \
  --predicted-size 36 \
  src/PI-CASA/2_way_model/benchmark_1.model \
  > output/2way/stdout.log 2> output/2way/stderr.log
```

3-way example:

```bash
mkdir -p output/3way
./PI-CASA -s 1 -y 1000 \
  -c src/PI-CASA/3_way_model/benchmark_1.constraints \
  -o output/PI-CASA/3way/result.txt \
  --predicted-size 258 \
  src/3_way_model/benchmark_1.model \
  > output/3way/stdout.log 2> output/3way/stderr.log
```

`-s` sets the random seed, `-y` sets the budget in CPU seconds, and `--predicted-size` supplies the precomputed positive integer initialization size. To run another benchmark, replace the model, constraints, initialization size, and output paths.

When a result is generated successfully, the first line of `result.txt` records the generation time in CPU seconds, the second line gives the covering array size, and each subsequent line contains one test case.

