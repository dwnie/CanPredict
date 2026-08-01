package parallel;

import engine.Main;
import ftchecker.Tuple;
import struct.CombinationPosition;
import struct.Element;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.concurrent.ForkJoinPool;

public class NeighborEvalParallel2 implements Serializable,
        NeighborEval {
    private static final long serialVersionUID = 1L;

    private final int row;

    private final int column;

    private final int newValue;

    private final Element[] neighbors;

    private final int no;

    private final ArrayList<Tuple> newMfts;

    private final ArrayList<CombinationPosition> unCoveredList;

    private final ArrayList<CombinationPosition> onceCoveredList;

    private final Tuple newCheckedTuple;

    public NeighborEvalParallel2(int row, int column, int newValue, Element[] neighbors, int no) {
        this.row = row;
        this.column = column;
        this.newValue = newValue;
        this.neighbors = neighbors;
        this.no = no;

        newMfts = Main.checker.getMFTbyParam(column, newValue);
        unCoveredList = Main.coveredCombination.getUncoveredList();
        onceCoveredList = Main.coveredCombination.getOnceCoveredListByVar(column, Main.coverageArray[row][column]);

        int[] newCheckedArray = new int[2 * Main.paraNum];
        for (int k = 0; k < Main.paraNum; k++) {
            newCheckedArray[2 * k] = k;
            newCheckedArray[2 * k + 1] = Main.coverageArray[row][k];
        }
        newCheckedArray[2 * column + 1] = newValue;
        newCheckedTuple = new Tuple(newCheckedArray);
    }

    public void caculEffcAndTabu() {

        int len = newMfts.size() + unCoveredList.size() + onceCoveredList.size();

        ForkJoinPool pool = new ForkJoinPool();

        Result result = pool.invoke(
                new FitnessCalculateTask2(row, column, newValue, newMfts, unCoveredList, onceCoveredList,
                        newCheckedTuple, 0, len));

        pool.shutdown();

        int mfts = result.getFitness_mfts();

        int uncoveredCombs = result.getFitness_uncoveredCombs();

        boolean istabu = Main.tabuList.isTabu(row, column, newValue);

        neighbors[no] = new Element(row, column, newValue, mfts, uncoveredCombs, istabu);

    }
}
