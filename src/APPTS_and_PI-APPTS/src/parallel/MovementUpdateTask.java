package parallel;

import engine.Main;

import java.util.concurrent.RecursiveAction;

public class MovementUpdateTask extends RecursiveAction {

    private final int THRESHOLD;

    private final int row;

    private final int column;

    private final int newValue;

    private final int[][] tuple_array;
    private final int start, end;

    public MovementUpdateTask(int row, int column, int newValue, int[][] tuple_array, int start, int end) {
        this.row = row;
        this.column = column;
        this.newValue = newValue;
        this.tuple_array = tuple_array;
        this.start = start;
        this.end = end;
        int len = tuple_array.length;
        this.THRESHOLD = Math.max(10, (2*len) / (Runtime.getRuntime().availableProcessors() * 4));
    }

    @Override
    protected void compute() {
        if (end - start <= THRESHOLD) {

            int[] value_tuple = new int[Main.t_way];
            int len = tuple_array.length;

            for (int k = start; k < end; k++) {
                if (k < len) {

                    for (int i = 0; i < Main.t_way; i++) {
                        value_tuple[i] = Main.coverageArray[row][tuple_array[k][i]];
                    }

                    int combinationRow = Main.paraCombination.getRowNum(tuple_array[k]);

                    int combinationColumn = Main.paraValueCombination.getColumnNum(tuple_array[k], value_tuple);

                    Main.coveredCombination.dec(combinationRow, combinationColumn);
                } else {

                    for (int i = 0; i < Main.t_way; i++) {
                        if (tuple_array[k - len][i] == column)
                            value_tuple[i] = newValue;
                        else
                            value_tuple[i] = Main.coverageArray[row][tuple_array[k - len][i]];
                    }

                    int combinationRow = Main.paraCombination.getRowNum(tuple_array[k - len]);

                    int combinationColumn = Main.paraValueCombination.getColumnNum(tuple_array[k - len], value_tuple);

                    Main.coveredCombination.inc(combinationRow, combinationColumn);
                }
            }
        } else {

            int middle = (start + end) / 2;
            MovementUpdateTask task1 = new MovementUpdateTask(row, column, newValue, tuple_array, start, middle);
            MovementUpdateTask task2 = new MovementUpdateTask(row, column, newValue, tuple_array, middle, end);

            invokeAll(task1, task2);
        }
    }
}
