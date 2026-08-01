package parallel;

import engine.Main;
import ftchecker.Tuple;
import struct.CombinationPosition;

import java.util.ArrayList;
import java.util.concurrent.RecursiveTask;

// Fork/Join reduction for the complete move effect: removed and introduced MFTs,
// newly covered interactions, and interactions that lose their last coverage.
public class FitnessCalculateTask extends RecursiveTask<Result> {

    private final int THRESHOLD;

    private final int row;

    private final int column;

    private final int newValue;

    private final ArrayList<Tuple> oldMfts;

    private final ArrayList<Tuple> newMfts;

    private final ArrayList<CombinationPosition> unCoveredList;

    private final ArrayList<CombinationPosition> onceCoveredList;

    private final Tuple oldCheckedTuple;

    private final Tuple newCheckedTuple;
    private final int start, end;

    public FitnessCalculateTask(int row, int column, int newValue, ArrayList<Tuple> oldMfts, ArrayList<Tuple> newMfts,
            ArrayList<CombinationPosition> unCoveredList, ArrayList<CombinationPosition> onceCoveredList,
            Tuple oldCheckedTuple, Tuple newCheckedTuple, int start, int end) {
        this.row = row;
        this.column = column;
        this.newValue = newValue;
        this.oldMfts = oldMfts;
        this.newMfts = newMfts;
        this.unCoveredList = unCoveredList;
        this.onceCoveredList = onceCoveredList;
        this.oldCheckedTuple = oldCheckedTuple;
        this.newCheckedTuple = newCheckedTuple;

        this.start = start;
        this.end = end;
        int len1 = this.oldMfts.size();
        int len2 = this.newMfts.size();
        int len3 = this.unCoveredList.size();
        int len4 = this.onceCoveredList.size();
        this.THRESHOLD = Math.max(10, (len1 + len2 + len3 + len4) / (Runtime.getRuntime().availableProcessors() * 4));
    }

    @Override
    protected Result compute() {
        if (end - start <= THRESHOLD) {

            int fitness_mft = 0;
            int fitness_comb = 0;
            int len1 = this.oldMfts.size();
            int len2 = this.newMfts.size();
            int len3 = this.unCoveredList.size();
            for (int i = start; i < end; i++) {
                if (i < len1)
                    fitness_mft -= deletedMft(i);
                else if (i < len1 + len2)
                    fitness_mft += addedMft(i - len1);
                else if (i < len1 + len2 + len3)
                    fitness_comb -= deletedUncoveredComb(i - len1 - len2);
                else
                    fitness_comb += addedUncoveredComb(i - len1 - len2 - len3);
            }
            return new Result(fitness_mft, fitness_comb);
        } else {

            int mid = (start + end) / 2;
            FitnessCalculateTask leftTask = new FitnessCalculateTask(row, column, newValue, oldMfts, newMfts,
                    unCoveredList, onceCoveredList, oldCheckedTuple, newCheckedTuple, start, mid);
            FitnessCalculateTask rightTask = new FitnessCalculateTask(row, column, newValue, oldMfts, newMfts,
                    unCoveredList, onceCoveredList, oldCheckedTuple, newCheckedTuple, mid, end);

            leftTask.fork();

            Result rightResult = rightTask.compute();
            Result leftResult = leftTask.join();

            return Result.merge(leftResult, rightResult);
        }
    }

    private int deletedMft(int i) {

        if (oldCheckedTuple.covers(oldMfts.get(i)))
            return 1;
        else
            return 0;
    }

    private int addedMft(int i) {

        if (newCheckedTuple.covers(newMfts.get(i)))
            return 1;
        else
            return 0;
    }

    private int deletedUncoveredComb(int i) {
        int[] tuple = new int[Main.t_way];
        int[] value_tuple = new int[Main.t_way];
        CombinationPosition element = unCoveredList.get(i);

        int combinationRow = element.combinationRow;
        Main.paraCombination.gett_tuple(combinationRow, tuple);

        int combinationColumn = element.combinationColumn;
        Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

        boolean flag = true;

        for (int j = 0; j < Main.t_way && flag; j++) {
            if (tuple[j] != column) {
                if (value_tuple[j] != Main.coverageArray[row][tuple[j]])
                    flag = false;
            } else {
                if (value_tuple[j] != newValue)
                    flag = false;
            }
        }

        if (flag)
            return 1;
        else
            return 0;
    }

    private int addedUncoveredComb(int i) {
        int[] tuple = new int[Main.t_way];
        int[] value_tuple = new int[Main.t_way];
        CombinationPosition element = onceCoveredList.get(i);

        int combinationRow = element.combinationRow;
        Main.paraCombination.gett_tuple(combinationRow, tuple);

        int combinationColumn = element.combinationColumn;
        Main.paraValueCombination.gett_tuple(tuple, combinationColumn, value_tuple);

        boolean flag = true;

        for (int j = 0; j < Main.t_way && flag; j++) {
            if (value_tuple[j] != Main.coverageArray[row][tuple[j]]) {
                flag = false;
                break;
            }
        }

        if (flag)
            return 1;
        else
            return 0;
    }
}
