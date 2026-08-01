# APPTS and PI-APPTS

This package provides executable implementations of APPTS and PI-APPTS for constrained covering-array generation.

## Requirements

- Java 11 or later.
- `APPTS.jar` and `ipog-ft.jar` in the project directory.
- Input `.model` and `.constraints` files in the `model/` directory.

Run all commands from the project directory because the program resolves these files through relative paths.

## Command

```bash
java -jar APPTS.jar <cutoffTime> <model> <constraints> <t_way> [<iterNum>] [<startSize>]
```

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `cutoffTime` | Maximum execution time in seconds. | Required |
| `model` | Name of the model file under `model/`. | Required |
| `constraints` | Name of the constraint file under `model/`. | Required |
| `t_way` | Interaction strength of the covering array. | Required |
| `iterNum` | Maximum number of consecutive non-improving iterations before diversification or backtracking. | `5000` |
| `startSize` | Predicted initial covering-array size. Supplying this parameter enables PI-APPTS; omitting it runs standard APPTS. | Disabled |

## Examples

Run standard APPTS:

```bash
java -jar APPTS.jar 300 benchmark_1.model benchmark_1.constraints 2
```

Run APPTS with an explicit non-improvement limit:

```bash
java -jar APPTS.jar 300 benchmark_1.model benchmark_1.constraints 2 5000
```

Run PI-APPTS with a predicted initial size of 48:

```bash
java -jar APPTS.jar 300 benchmark_1.model benchmark_1.constraints 2 5000 48
```

## Output

The program prints progress to the terminal and writes the best covering array found within the cutoff time to `result.txt`.
